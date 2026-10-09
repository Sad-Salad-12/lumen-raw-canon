"""Photo-to-Gemini protocol, conservative verdicts, resume and album/UI behavior."""
import copy
import io
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QMessageBox

from lumen import culling, gemini, nl_edit
from lumen.culling_panel import CullingDialog
from lumen.library import dirty, new_document
from lumen.scheduler import Activity as A
from test_gemini import server, service  # noqa: F401
from test_ui import app, window, wait_until  # noqa: F401


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch):
    monkeypatch.setattr(nl_edit, 'load_settings', lambda: nl_edit.default_settings())
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Discard)


@pytest.fixture
def photos(tmp_path):
    paths = []
    for index in range(8):
        path = tmp_path / f'frame-{index:02d}.png'
        yy, xx = np.mgrid[:100, :150]
        pixels = np.stack([xx * 1.5, yy * 2, np.full_like(xx, 40 + index)], -1).astype(np.uint8)
        Image.fromarray(pixels).save(path)
        paths.append(str(path))
    return paths


def row(id, verdict='keep', confidence=.96):
    return dict(id=id, verdict=verdict, score=85, confidence=confidence,
                reason='Subject is clear; a useful expression.', issues=[], duplicate_of='')


def batch(count=2):
    return [dict(id=f'p{i:04d}', name=f'photo-{i}.CR3', group='0') for i in range(count)]


def fake_assessor(monkeypatch, cancel=None, error_at=None):
    calls = []

    def complete(service, messages, **kwargs):
        ids = [json.loads(label)['id'] for label, _ in kwargs['images']]
        calls.append(ids)
        assert len(ids) <= 6 and kwargs['schema']['properties']['photos']['minItems'] == len(ids)
        if error_at == len(calls):
            raise nl_edit.ServiceError('HTTP 429 quota', 429)
        if cancel:
            cancel.set()
        return json.dumps({'photos': [row(id) for id in ids]})

    monkeypatch.setattr(gemini, 'complete', complete)
    return calls


def test_native_wire_sends_multiple_labelled_jpegs_and_no_original(server, photos):
    server['data'] = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
        {'text': json.dumps({'photos': [row('p0000'), row('p0001', 'reject', .97)]})}]}}]}
    original = [Path(p).read_bytes() for p in photos[:2]]
    events = []
    report = culling.run(photos[:2], {}, service(server), culling.options(), progress=events.append)
    assert report['requests'] == 1 and not report['errors']
    assert [r['verdict'] for r in report['records'].values()] == ['keep', 'reject']
    _, headers, payload = server['requests'][0]
    assert headers['X-Goog-Api-Key'] == 'test-key'
    parts = payload['contents'][0]['parts']
    assert len([p for p in parts if 'inlineData' in p]) == 2
    import base64
    for index in (1, 3):
        assert json.loads(parts[index]['text'])['id'] in ('p0000', 'p0001')
        pic = Image.open(io.BytesIO(base64.b64decode(parts[index + 1]['inlineData']['data'])))
        assert pic.format == 'JPEG' and max(pic.size) <= 1600
    assert all(p not in json.dumps(payload) for p in photos)
    assert payload['generationConfig']['responseFormat']['text']['schema']['properties']['photos']['maxItems'] == 2
    assert [Path(p).read_bytes() for p in photos[:2]] == original
    assert len([e for e in events if 'record' in e]) == 2


def test_low_confidence_and_unverified_duplicates_go_to_review():
    rows = [row('p0000', 'reject', .89), row('p0001', confidence=.8)]
    result = culling.validate_reply(json.dumps({'photos': rows}), batch(), .85)
    assert result['p0000']['verdict'] == 'review'
    assert result['p0001']['verdict'] == 'review'
    rows[0].update(confidence=.99, issues=['duplicate'])
    result = culling.validate_reply(json.dumps({'photos': rows}), batch(), .85)
    assert result['p0000']['verdict'] == 'review'


def test_verified_duplicate_requires_confident_kept_target():
    rows = [row('p0000'), row('p0001', 'reject', .97)]
    rows[1].update(duplicate_of='p0000', issues=['duplicate'])
    result = culling.validate_reply(json.dumps({'photos': rows}), batch(), .90)
    assert result['p0001']['verdict'] == 'reject' and result['p0001']['duplicate_of'] == 'photo-0.CR3'
    rows[0]['confidence'] = .80
    result = culling.validate_reply(json.dumps({'photos': rows}), batch(), .90)
    assert all(r['verdict'] == 'review' for r in result.values())
    rows[0].update(confidence=.99, issues=['duplicate'])
    result = culling.validate_reply(json.dumps({'photos': rows}), batch(), .90)
    assert all(r['verdict'] == 'review' for r in result.values())


@pytest.mark.parametrize('change', [
    lambda r: r.pop(),
    lambda r: r[1].update(id='unknown'),
    lambda r: r[1].update(id='p0000'),
    lambda r: r[0].update(confidence=float('nan')),
    lambda r: r[0].update(score=True),
    lambda r: r[0].update(reason=''),
    lambda r: r[0].update(issues=['imaginary']),
    lambda r: r[0].update(duplicate_of='p0000'),
    lambda r: r[0].update(duplicate_of='p0001'),
])
def test_invalid_assessments_cannot_become_rejections(change):
    rows = [row('p0000'), row('p0001', 'review')]
    change(rows)
    with pytest.raises(ValueError):
        culling.validate_reply(json.dumps({'photos': rows}), batch(), .9)


def test_cross_group_links_and_duplicate_cycles_are_invalid():
    rows = [row('p0000'), row('p0001', 'reject')]
    rows[1]['duplicate_of'] = 'p0000'
    items = batch()
    items[1]['group'] = '1'
    with pytest.raises(ValueError):
        culling.validate_reply(json.dumps({'photos': rows}), items, .9)
    rows[0]['duplicate_of'] = 'p0001'
    rows[1]['verdict'] = 'keep'
    with pytest.raises(ValueError):
        culling.validate_reply(json.dumps({'photos': rows}), batch(), .9)


def test_cache_reuse_and_invalidation_of_preferences_model_and_photo(monkeypatch, server, photos):
    calls = fake_assessor(monkeypatch)
    chosen = photos[:2]
    config = culling.options(criteria='Allow night noise')
    first = culling.run(chosen, {}, service(server), config)
    assert len(calls) == 1
    second = culling.run(chosen, first['records'], service(server), config)
    assert len(calls) == 1 and second['cached'] == 2 and second['requests'] == 0
    changed = culling.run(chosen, second['records'], service(server), culling.options(criteria='Eyes sharp'))
    assert len(calls) == 2 and changed['cached'] == 0
    changed_model = culling.run(chosen, changed['records'], service(server, model='gemini-test'), culling.options(criteria='Eyes sharp'))
    assert len(calls) == 3 and changed_model['cached'] == 0
    Image.new('RGB', (150, 100), 'white').save(chosen[0])
    changed_photo = culling.run(chosen, changed_model['records'], service(server, model='gemini-test'), culling.options(criteria='Eyes sharp'))
    assert len(calls) == 4 and changed_photo['cached'] == 0


def test_cancel_drops_inflight_result_and_prevents_next_request(monkeypatch, server, photos):
    event = threading.Event()
    calls = fake_assessor(monkeypatch, cancel=event)
    result = culling.run(photos, {}, service(server), culling.options(), cancel=event)
    assert result['cancelled'] and len(calls) == 1 and not result['records']


def test_cancel_after_first_batch_preserves_completed_records(monkeypatch, server, photos):
    cancel = threading.Event()
    calls = []
    def complete(service, messages, **kwargs):
        ids = [json.loads(label)['id'] for label, _ in kwargs['images']]
        calls.append(ids)
        if len(calls) == 2:
            cancel.set()
        return json.dumps({'photos': [row(id) for id in ids]})
    monkeypatch.setattr(gemini, 'complete', complete)
    result = culling.run(photos, {}, service(server), culling.options(), cancel=cancel)
    assert result['cancelled'] and len(result['records']) == 6 and len(calls) == 2


def test_raw_preview_uses_camera_development_and_caps_image_size(monkeypatch):
    from lumen import engine
    calls = []
    linear = np.full((1200, 2000, 3), .1, np.float32)
    def load(path, limit):
        calls.append((path, limit))
        resized = engine.resize_limit(linear, limit)
        return resized, dict(raw=True, develop=dict(mode='camera', curve=[[0, 0], [.1, .4], [1, 1]]))
    monkeypatch.setattr(engine, 'load_image', load)
    encoded, evidence = culling.preview('Canon.CR3')
    import base64
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert calls == [('Canon.CR3', 1600)] and max(image.size) == 1600
    # Camera curve lifts .1 to .4 in linear light before the display transform.
    expected = float(engine.to_srgb(np.array(.4, np.float32)))
    assert abs(np.asarray(image).mean() / 255 - expected) < .01
    assert evidence['metrics']['sharpness'] == 0


def test_quota_failure_preserves_completed_batch_and_resume_retries_only_failed(monkeypatch, server, photos):
    calls = fake_assessor(monkeypatch, error_at=2)
    result = culling.run(photos, {}, service(server), culling.options())
    assert len(calls) == 2 and len(result['records']) == 8
    assert sum(r['verdict'] == 'keep' for r in result['records'].values()) == 6
    assert sum(bool(r.get('error')) for r in result['records'].values()) == 2
    resumed = culling.run(photos, result['records'], service(server), culling.options())
    assert len(calls) == 3 and resumed['cached'] == 6 and resumed['requests'] == 1
    assert all(r['verdict'] == 'keep' for r in resumed['records'].values())


def test_decode_failure_does_not_block_other_photos_or_reject_bad_input(monkeypatch, server, photos, tmp_path):
    calls = fake_assessor(monkeypatch)
    bad = tmp_path / 'corrupt.CR3'
    bad.write_bytes(b'not a RAW photograph')
    result = culling.run([str(bad), *photos[:2]], {}, service(server), culling.options())
    assert result['records'][str(bad)]['verdict'] == 'review'
    assert result['records'][str(bad)]['error'] and len(calls) == 1
    assert all(result['records'][p]['verdict'] == 'keep' for p in photos[:2])


def test_manual_overrides_are_preserved_even_with_forced_analysis(monkeypatch, server, photos):
    calls = fake_assessor(monkeypatch)
    old = {p: {'manual': 'reject'} for p in photos}
    before = copy.deepcopy(old)
    result = culling.run(photos, old, service(server), culling.options(force=True))
    assert result['manual'] == 8 and not result['records'] and not calls and old == before


def test_streaming_starts_before_full_shoot_and_decodes_each_new_photo_once(monkeypatch, server, photos):
    real_preview = culling.preview
    decoded, calls = [], []
    def preview(path):
        decoded.append(path)
        return real_preview(path)
    def complete(service, messages, **kwargs):
        ids = [json.loads(label)['id'] for label, _ in kwargs['images']]
        calls.append(len(decoded))
        return json.dumps({'photos': [row(id) for id in ids]})
    monkeypatch.setattr(culling, 'preview', preview)
    monkeypatch.setattr(gemini, 'complete', complete)
    result = culling.run(photos, {}, service(server), culling.options())
    assert calls[0] < len(photos) and len(decoded) == len(photos) and len(result['records']) == len(photos)
    culling.run(photos, result['records'], service(server), culling.options())
    assert len(decoded) == len(photos) and len(calls) == 2


def test_no_key_or_compatible_endpoint_never_starts_a_request(monkeypatch, server, photos):
    calls = fake_assessor(monkeypatch)
    for changes in ({'key': ''}, {'base_url': server['url'] + '/openai'}):
        with pytest.raises(ValueError):
            culling.run(photos, {}, service(server, **changes), culling.options())
    assert not calls


def test_file_changed_during_request_is_never_marked_keep(monkeypatch, server, photos):
    def complete(service, messages, **kwargs):
        Image.new('RGB', (20, 20), 'red').save(photos[0])
        return json.dumps({'photos': [row('p0000')]})
    monkeypatch.setattr(gemini, 'complete', complete)
    result = culling.run(photos[:1], {}, service(server), culling.options())
    assert result['records'][photos[0]]['verdict'] == 'review'
    assert '变化' in result['records'][photos[0]]['error']


def test_group_boundaries_respect_capture_times_and_window_limit():
    items = [dict(id=str(i), prepared=dict(visual=15, aspect=1.5, date='2026-10-08 10:00:00')) for i in range(14)]
    items[-1]['prepared']['date'] = '2026-10-08 10:01:00'
    groups = list(culling.batches(items))
    assert [len(group) for group in groups] == [6, 6, 2]
    assert items[-2]['group'] != items[-1]['group']
    assert all(len(group) <= 6 for group in groups)


def test_album_record_validation_handles_old_or_malformed_records():
    assert culling.clean_record(None) == {} and culling.clean_record({'verdict': 'reject'}) == {}
    assert culling.album_options({'confidence': 'bad'}) == culling.options()
    assert culling.clean_record({'manual': 'keep'}) == {'manual': 'keep'}
    assert culling.clean_record(dict(version=1, verdict='reject', score=1, confidence=float('nan'))) == {}
    assert culling.clean_record(dict(version=1, verdict='reject', score=1, confidence=1, prepared={'metrics': []}))['verdict'] == 'reject'
    document = new_document('/imaginary.CR3')
    assert not dirty(document)
    document['culling'] = {'manual': 'keep'}
    assert dirty(document)


def test_ui_analysis_filter_export_selection_override_undo_and_album_roundtrip(window, monkeypatch, server, photos, tmp_path):
    w = window
    w.add_documents(photos[:2])
    wait_until(lambda: not w.jobs)
    def complete(service, messages, **kwargs):
        ids = [json.loads(label)['id'] for label, _ in kwargs['images']]
        return json.dumps({'photos': [row(id, 'keep' if i == 0 else 'reject') for i, id in enumerate(ids)]})
    monkeypatch.setattr(gemini, 'complete', complete)
    w.nl_settings['providers']['gemini'].update(base_url=server['url'], key='test-key')
    dialog = CullingDialog(w)
    dialog.genre.setCurrentIndex(1)
    dialog.criteria.setText('保留自然的表情')
    dialog.start()
    assert dialog.running and w.work.busy()
    wait_until(lambda: not dialog.running and not w.jobs)
    assert 'API 请求 1' in dialog.summary.text()
    assert len(w.cull_history) == 1 and all(dirty(d) for d in w.documents.values())
    w.select_culling_keeps()
    kept = w.selected_paths()
    assert len(kept) == 1
    assert all(not w.film_item(p).isSelected() for p in photos[:2])
    w.film_item(photos[0]).setSelected(True)
    assert photos[0] not in w.selected_paths()
    w.mark_culling([photos[0]], 'keep')
    assert culling.status(w.documents[photos[0]]['culling']) == 'keep'
    w.undo_culling()
    assert culling.status(w.documents[photos[0]]['culling']) == 'reject'
    w.mark_culling([photos[0]], 'keep')
    album = tmp_path / 'test.lumenalbum'
    assert w.save_album(path=album)
    assert not any(dirty(d) for d in w.documents.values())
    saved = json.loads(album.read_text())
    assert saved['documents'][1]['culling']['manual'] == 'keep'
    w.open_album(album)
    wait_until(lambda: not w.jobs)
    assert w.documents[photos[0]]['culling']['manual'] == 'keep'
    assert not w.cull_history and not any(dirty(d) for d in w.documents.values())
    assert w.cull_filter.currentData() == 'all'
    assert w.cull_preferences['genre'] == 'portrait' and w.cull_preferences['criteria'] == '保留自然的表情'
    resumed = CullingDialog(w)
    resumed.start()
    wait_until(lambda: not resumed.running and not w.jobs)
    # Pinning a manual decision changes comparison context for the other frames.
    assert 'API 请求 1' in resumed.summary.text()
    assert w.documents[photos[0]]['culling']['manual'] == 'keep'
    resumed.close()
    assert w.save_album(path=album)
    w.open_album(album)
    wait_until(lambda: not w.jobs)
    cached = CullingDialog(w)
    cached.start()
    wait_until(lambda: not cached.running and not w.jobs)
    assert 'API 请求 0' in cached.summary.text() and '复用 2' in cached.summary.text()
    assert not w.cull_history and not any(dirty(d) for d in w.documents.values())
    cached.close()
    for document in saved['documents']:
        document.pop('culling')
    album.write_text(json.dumps(saved))
    w.open_album(album)
    wait_until(lambda: not w.jobs)
    assert all(d['culling'] == {} for d in w.documents.values())
    dialog.close()


def test_ui_selected_scope_and_cancel_wait_for_active_request(window, monkeypatch, server, photos):
    w = window
    w.add_documents(photos[:2])
    wait_until(lambda: not w.jobs)
    w.filmstrip.clearSelection()
    w.film_item(photos[1]).setSelected(True)
    started, release = threading.Event(), threading.Event()
    def complete(service, messages, **kwargs):
        assert len(kwargs['images']) == 1
        id = json.loads(kwargs['images'][0][0])['id']
        started.set()
        release.wait(5)
        return json.dumps({'photos': [row(id)]})
    monkeypatch.setattr(gemini, 'complete', complete)
    w.nl_settings['providers']['gemini'].update(base_url=server['url'], key='test-key')
    dialog = CullingDialog(w)
    dialog.scope.setCurrentIndex(1)
    dialog.show()
    dialog.start()
    wait_until(started.is_set)
    dialog.reject()
    assert dialog.isVisible() and dialog.running and dialog.cancel_event.is_set()
    assert not w.work.can_start(A.EXPORTING)
    release.set()
    wait_until(lambda: not dialog.running and not w.jobs)
    assert '已停止' in dialog.summary.text() and all(not d['culling'] for d in w.documents.values())
    assert not w.cull_history
    dialog.reject()


def test_ui_manual_review_table_and_culling_undo_do_not_change_edit_history(window, photos):
    w = window
    w.add_documents(photos[:2])
    wait_until(lambda: not w.jobs)
    history = copy.deepcopy(w.edits)
    dialog = CullingDialog(w)
    dialog.table.selectRow(1)
    dialog.mark('keep')
    assert w.documents[photos[0]]['culling']['manual'] == 'keep'
    assert dialog.table.item(1, 2).text() == '推荐保留 · 人工'
    dialog.undo()
    assert w.documents[photos[0]]['culling'] == {} and w.edits == history
    dialog.close()

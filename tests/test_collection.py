"""Collection-first workflow, one shared style request and reversible initial development."""
import copy
import json
import threading
from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox

from lumen import collection, engine, gemini, model, nl_edit
from lumen.app import MainWindow
from lumen.library import new_document, dirty
from test_culling import photos, isolate_settings  # noqa: F401
from test_gemini import server, service  # noqa: F401
from test_ui import app, window, wait_until  # noqa: F401


def documents(paths, verdict='keep'):
    result = {p: new_document(p) for p in paths}
    for d in result.values():
        d['culling'] = {'manual': verdict}
    return result


def apply(docs, report):
    for p, change in report['changes'].items():
        docs[p].update(copy.deepcopy(change))


def fake_style(monkeypatch, offsets=None, cancel=None, error=None):
    calls = []
    def complete(service, messages, **kw):
        calls.append((messages, kw))
        if cancel:cancel.set()
        if error:raise error
        return json.dumps(dict(explanation='整组保持自然暖调', adjustments=offsets or {'temperature': 10, 'vibrance': 8}))
    monkeypatch.setattr(gemini, 'complete', complete)
    return calls


def test_local_only_keepers_and_preserves_individual_geometry_and_metadata(photos, monkeypatch):
    docs = documents(photos[:4])
    docs[photos[1]]['culling'] = {'manual': 'review'}
    docs[photos[2]]['culling'] = {'manual': 'reject'}
    docs[photos[3]]['culling'] = {}
    d = docs[photos[0]]
    d['initialized'] = True
    d['edits']['crop'] = [.1, .2, .8, .7]
    d['edits']['rotation'] = 3
    d['edits']['wb_gain'] = [1.1, 1, .9]
    d['edits']['masks'] = [model.new_mask('linear', 1)]
    d['edits']['retouch'] = [dict(kind='heal', enabled=True, points=[[.4, .4]], radius=.02, opacity=100, feather=50)]
    d['edits']['white_balance'] = dict(camera_kelvin=5200., kelvin=5000., estimated=False)
    original = copy.deepcopy(docs)
    source_bytes = {p: Path(p).read_bytes() for p in docs}
    monkeypatch.setattr(gemini, 'complete', lambda *a, **k: pytest.fail('Local adjustment must not send photos'))
    report = collection.run(docs, collection.settings())
    assert not report['errors'] and report['requests'] == 0 and report['skipped'] == 3
    assert list(report['changes']) == photos[:1]
    target = report['changes'][photos[0]]['edits']
    for key in d['edits']:
        if key not in ('adjustments', 'lens'):
            assert target[key] == d['edits'][key], key
    assert abs(target['adjustments']['exposure']) <= .375
    assert docs == original
    assert source_bytes == {p: Path(p).read_bytes() for p in docs}


def test_local_cache_preference_rerun_no_compounding_and_manual_edits_skip(photos, monkeypatch):
    docs = documents(photos[:2])
    first = collection.run(docs, collection.settings())
    apply(docs, first)
    loader = engine.load_image
    monkeypatch.setattr(engine, 'load_image', lambda *a: pytest.fail('Cached photos must not decode'))
    cached = collection.run(docs, collection.settings())
    assert cached['cached'] == 2 and not cached['changes']
    monkeypatch.setattr(engine, 'load_image', loader)
    twice = collection.run(docs, collection.settings(strength=1))
    independent = collection.run(documents(photos[:2]), collection.settings(strength=1))
    assert [v['edits'] for v in twice['changes'].values()] == [v['edits'] for v in independent['changes'].values()]
    docs[photos[0]]['edits']['adjustments']['exposure'] += .1
    skipped = collection.run(docs, collection.settings())
    assert skipped['skipped'] == 1 and skipped['cached'] == 1
    overwrite = collection.run(docs, collection.settings(overwrite=True))
    assert photos[0] in overwrite['changes']


def test_one_gemini_request_for_entire_collection_and_distributed_sample(monkeypatch, photos, server):
    calls = fake_style(monkeypatch)
    docs = documents(photos)
    report = collection.run(docs, collection.settings(mode='gemini'), service(server), genre='travel')
    assert len(calls) == report['requests'] == 1 and len(report['changes']) == 8 and not report['errors']
    messages, kw = calls[0]
    ids = [json.loads(label)['id'] for label, _ in kw['images']]
    assert ids == ['sample-0', 'sample-1', 'sample-3', 'sample-4', 'sample-6', 'sample-7']
    assert 'travel' in messages[1]['content'] and kw['schema']['additionalProperties'] is False
    assert all(p not in json.dumps(messages) + json.dumps(kw) for p in photos)
    for value in report['changes'].values():
        assert value['edits']['adjustments']['temperature'] == 5
        assert value['edits']['adjustments']['vibrance'] == 4
    apply(docs, report)
    cached = collection.run(docs, collection.settings(mode='gemini'), service(server))
    # Genre changes the policy.
    assert cached['requests'] == 1
    apply(docs, cached)
    assert collection.run(docs, collection.settings(mode='gemini'), service(server))['cached'] == 8
    smaller = {p: d for p, d in docs.items() if p != photos[-1]}
    changed = collection.run(smaller, collection.settings(mode='gemini'), service(server))
    assert changed['requests'] == 1 and len(changed['changes']) == 7


@pytest.mark.parametrize('offsets', [{'crop': [0, 0, 1, 1]}, {'exposure': 9}, {'temperature': float('nan')}, {'contrast': True}])
def test_invalid_shared_look_never_changes_recipes(monkeypatch, photos, server, offsets):
    fake_style(monkeypatch, offsets)
    docs = documents(photos[:2])
    before = copy.deepcopy(docs)
    report = collection.run(docs, collection.settings(mode='gemini'), service(server))
    assert len(report['errors']) == 2
    assert all('edits' not in v and v['initial_adjustment']['status'] == 'error' for v in report['changes'].values())
    assert docs == before


@pytest.mark.parametrize('error', [None, RuntimeError('Cancelled network response')])
def test_cancel_discards_inflight_shared_look(monkeypatch, photos, server, error):
    cancel = threading.Event()
    calls = fake_style(monkeypatch, cancel=cancel, error=error)
    result = collection.run(documents(photos), collection.settings(mode='gemini'), service(server), cancel=cancel)
    assert result['cancelled'] and len(calls) == 1 and not result['changes']


def test_decode_failure_isolated_and_first_valid_sample_used(monkeypatch, photos, server):
    loader = engine.load_image
    def load(path, size):
        if path in photos[:1]:raise OSError('Corrupt preview')
        return loader(path, size)
    monkeypatch.setattr(engine, 'load_image', load)
    calls = fake_style(monkeypatch)
    result = collection.run(documents(photos[:2]), collection.settings(mode='gemini'), service(server))
    assert len(result['errors']) == 1 and len(calls) == 1
    assert json.loads(calls[0][1]['images'][0][0])['id'] == 'sample-1'
    assert result['changes'][photos[1]]['initial_adjustment']['status'] == 'adjusted'


def test_file_change_during_shared_request_cannot_apply_recipe(monkeypatch, photos, server):
    from PIL import Image
    def complete(*args, **kwargs):
        Image.new('RGB', (10, 10), 'white').save(photos[0])
        return json.dumps(dict(explanation='Natural', adjustments={'temperature': 10}))
    monkeypatch.setattr(gemini, 'complete', complete)
    result = collection.run(documents(photos[:2]), collection.settings(mode='gemini'), service(server))
    assert 'edits' not in result['changes'][photos[0]]
    assert 'edits' in result['changes'][photos[1]]


def test_native_wire_one_request_contains_sample_not_originals(server, photos):
    response = json.dumps(dict(explanation='Natural shared colour', adjustments={'temperature': 6}))
    server['data'] = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': response}]}}]}
    result = collection.run(documents(photos), collection.settings(mode='gemini'), service(server))
    assert not result['errors'] and result['requests'] == 1 and len(result['changes']) == 8
    payload = server['requests'][0][2]
    assert len([p for p in payload['contents'][0]['parts'] if 'inlineData' in p]) == 6


def test_collection_default_import_remove_undo_preserve_order_and_originals(app, monkeypatch, photos, tmp_path):
    w = MainWindow()
    try:
        assert w.workspace_stack.currentWidget() is w.collection_page and w.source is None
        w.import_paths(photos[:4])
        wait_until(lambda: not w.jobs)
        assert w.source is None and w.workspace_stack.currentWidget() is w.collection_page
        before = {p: Path(p).read_bytes() for p in photos}
        w.mark_culling(photos[:1], 'keep')
        w.mark_culling(photos[1:3], 'reject')
        w.documents[photos[1]]['edits']['adjustments']['exposure'] = .3
        w.collection_page.filter.setCurrentIndex(1)  # Even hidden excludes are removed.
        w.collection_page.remove_excluded()
        assert list(w.documents) == [photos[0], photos[3]]
        w.import_paths(photos[4:5])
        wait_until(lambda: not w.jobs)
        w.undo_collection()
        wait_until(lambda: not w.jobs)
        assert list(w.documents) == photos[:5]
        assert w.documents[photos[1]]['edits']['adjustments']['exposure'] == .3
        assert before == {p: Path(p).read_bytes() for p in photos}
        w.remove_excluded()
        album = tmp_path / 'remaining.lumenalbum'
        assert w.save_album(path=album)
        w.open_album(album)
        wait_until(lambda: not w.jobs)
        assert list(w.documents) == [photos[0], photos[3], photos[4]] and w.source is None
        assert w.workspace_stack.currentWidget() is w.collection_page
    finally:
        monkeypatch.setattr(QMessageBox, 'question', lambda *a: QMessageBox.StandardButton.Discard)
        wait_until(lambda: not w.jobs)
        w.close()


def test_collection_initial_adjust_undo_album_cache_and_current_editor(window, photos, monkeypatch, tmp_path):
    w = window
    current = w.source_path
    w.add_documents(photos[:2])
    wait_until(lambda: not w.jobs)
    w.mark_culling([current, photos[0]], 'keep')
    w.mark_culling([photos[1]], 'reject')
    original = {p: copy.deepcopy(d['edits']) for p, d in w.documents.items()}
    w.show_collection()
    page = w.collection_page
    page.start_adjustment()
    wait_until(lambda: not page.running and not w.jobs)
    assert len(w.collection_history) == 1
    assert w.edits == w.documents[current]['edits'] != original[current]
    assert not w.documents[photos[1]]['initial_adjustment']
    assert all(w.documents[p]['initial_adjustment']['status'] == 'adjusted' for p in (current, photos[0]))
    # The hidden editor cannot overwrite the freshly applied collection recipe.
    w.stash_document()
    assert w.documents[current]['edits'] == w.edits
    album = tmp_path / 'developed.lumenalbum'
    assert w.save_album(path=album) and not any(dirty(d) for d in w.documents.values())
    w.undo_collection()
    assert w.edits == original[current]
    assert all(d['edits'] == original[p] for p, d in w.documents.items())
    assert not w.documents[current]['initial_adjustment'] and dirty(w.documents[current])
    w.open_album(album)
    wait_until(lambda: not w.jobs)
    assert not w.collection_history and not any(dirty(d) for d in w.documents.values())
    page.start_adjustment()
    wait_until(lambda: not page.running and not w.jobs)
    assert '复用 2' in page.summary.text() and not w.collection_history
    exported = []
    monkeypatch.setattr(w, 'batch_export', lambda: exported.extend(w.selected_paths()))
    w.export()
    assert exported == [current, photos[0]]
    saved = []
    monkeypatch.setattr(w, 'save_album', lambda: saved.append(True))
    w.save()
    assert saved == [True]


def test_retry_after_shared_error_reuses_original_initial_baseline(monkeypatch, photos, server):
    docs = documents(photos[:2])
    fake_style(monkeypatch, {'temperature': 8})
    first = collection.run(docs, collection.settings(mode='gemini'), service(server))
    apply(docs, first)
    fake_style(monkeypatch, error=nl_edit.ServiceError('Quota exhausted', 429))
    config = collection.settings(mode='gemini', instruction='Warmer')
    failed = collection.run(docs, config, service(server))
    apply(docs, failed)
    fake_style(monkeypatch, {'temperature': 12})
    retried = collection.run(docs, config, service(server))
    fresh = collection.run(documents(photos[:2]), config, service(server))
    assert not retried['errors']
    assert [v['edits'] for v in retried['changes'].values()] == [v['edits'] for v in fresh['changes'].values()]


def test_collection_ui_cull_remove_shared_develop_export_and_undo(app, monkeypatch, photos, server, tmp_path):
    calls = []
    def complete(service, messages, **kw):
        ids = [json.loads(label)['id'] for label, _ in kw['images']]
        calls.append(ids)
        if 'photos' in kw['schema']['properties']:
            from test_culling import row
            verdicts = ['keep', 'reject', 'review', 'keep', 'reject', 'keep']
            return json.dumps({'photos': [row(id, verdicts[int(id[1:]) % 6]) for id in ids]})
        return json.dumps(dict(explanation='统一暖调', adjustments={'temperature': 10}))
    monkeypatch.setattr(gemini, 'complete', complete)
    w = MainWindow()
    try:
        w.nl_settings['providers']['gemini'].update(base_url=server['url'], key='test-key')
        w.import_paths(photos)
        wait_until(lambda: not w.jobs)
        page = w.collection_page
        page.start()
        wait_until(lambda: not page.running and not w.jobs)
        assert w.source is None and len(calls) == 2
        keep = [p for p, d in w.documents.items() if d['culling']['verdict'] == 'keep']
        reject = [p for p, d in w.documents.items() if d['culling']['verdict'] == 'reject']
        assert len(keep) == 4 and len(reject) == 3
        page.filter.setCurrentIndex(3)
        assert not any(page.table.isRowHidden(page.rows[p]) for p in reject)
        page.remove_excluded()
        assert not any(p in w.documents for p in reject)
        page.adjust_mode.setCurrentIndex(1)
        page.adjust_instruction.setText('统一暖调')
        page.start_adjustment()
        wait_until(lambda: not page.running and not w.jobs)
        assert len(calls) == 3 and len(calls[-1]) == 4 and w.source is None
        assert all(w.documents[p]['edits']['adjustments']['temperature'] == 5 for p in keep)
        selected = []
        monkeypatch.setattr(w, 'batch_export', lambda: selected.extend(w.selected_paths()))
        page.export_keeps()
        assert selected == keep
        album = tmp_path / 'whole-shoot.lumenalbum'
        assert w.save_album(path=album)
        w.undo_collection()
        assert all(not w.documents[p]['initial_adjustment'] for p in keep)
        w.undo_collection()
        wait_until(lambda: not w.jobs)
        assert list(w.documents) == photos
        w.open_album(album)
        wait_until(lambda: not w.jobs)
        assert page.adjust_instruction.text() == '统一暖调' and page.adjust_mode.currentData() == 'gemini'
        page.start_adjustment()
        wait_until(lambda: not page.running and not w.jobs)
        assert len(calls) == 3 and '复用 4' in page.summary.text()
    finally:
        wait_until(lambda: not w.jobs)
        w.close()


def test_collection_selected_scope_uses_visible_workbench_rows(app, monkeypatch, photos, server):
    calls = []
    def complete(service, messages, **kw):
        from test_culling import row
        ids = [json.loads(label)['id'] for label, _ in kw['images']]
        calls.append(ids)
        return json.dumps({'photos': [row(id) for id in ids]})
    monkeypatch.setattr(gemini, 'complete', complete)
    w = MainWindow()
    try:
        w.nl_settings['providers']['gemini'].update(base_url=server['url'], key='test-key')
        w.import_paths(photos[:3])
        wait_until(lambda: not w.jobs)
        page = w.collection_page
        page.table.selectRow(1)
        page.scope.setCurrentIndex(1)
        page.start()
        wait_until(lambda: not page.running and not w.jobs)
        assert len(calls) == 1 and len(calls[0]) == 1
        assert w.documents[photos[1]]['culling']['verdict'] == 'keep'
        assert not w.documents[photos[0]]['culling'] and not w.documents[photos[2]]['culling']
    finally:
        wait_until(lambda: not w.jobs)
        w.close()

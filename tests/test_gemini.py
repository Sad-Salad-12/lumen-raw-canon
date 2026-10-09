"""Native Gemini wire protocol and its integration with the editor's recipe pipeline."""
import copy
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

from lumen import gemini, model, nl_edit
from test_ui import app  # noqa: F401


@pytest.fixture
def server():
    reply = {'explanation': '天空更蓝，阴影提亮。',
             'adjustments': {'shadows': 25}, 'hsl': {'blue': {'saturation': 12}},
             'masks': [{'region': 'top', 'adjustments': {'exposure': -.3}}]}
    state = {'requests': [], 'status': 200, 'data': {
        'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
            {'thought': True, 'text': 'Internal thought'}, {'text': json.dumps(reply)}]}}],
        'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 30, 'thoughtsTokenCount': 20}}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, data):
            body = json.dumps(data).encode()
            self.send_response(state['status'])
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['requests'].append((self.path, dict(self.headers), data))
            self.respond(state['data'])

        def do_GET(self):
            state['requests'].append((self.path, dict(self.headers), None))
            query = parse_qs(urlsplit(self.path).query)
            if query.get('pageToken') == ['next page']:
                self.respond({'models': [{'name': 'models/gemini-3.8-flash',
                                         'supportedGenerationMethods': ['generateContent']}]})
            else:
                self.respond({'nextPageToken': 'next page', 'models': [
                    {'name': 'models/gemini-3.1-pro-preview', 'supportedGenerationMethods': ['generateContent']},
                    {'name': 'models/gemini-embedding-001', 'supportedGenerationMethods': ['embedContent']},
                    {'name': 'models/gemini-2.5-flash-image', 'supportedGenerationMethods': ['generateContent']},
                    {'name': 'models/gemini-2.5-flash-native-audio', 'supportedGenerationMethods': ['generateContent']},
                    {'name': 'models/imagen-4', 'supportedGenerationMethods': ['predict']}]})

    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    state['url'] = f'http://127.0.0.1:{http.server_address[1]}/v1beta'
    yield state
    http.shutdown()
    http.server_close()
    thread.join()


def service(server, **changes):
    settings = nl_edit.default_settings()
    settings['provider'] = 'gemini'
    settings['reasoning'] = changes.pop('reasoning', 'off')
    settings['providers']['gemini'].update(base_url=server['url'], key='test-key')
    settings['providers']['gemini'].update(changes)
    return nl_edit.active(settings)


def test_photo_and_history_produce_an_applicable_recipe_without_thought_text(server):
    base = model.recipe()
    untouched = copy.deepcopy(base)
    messages = nl_edit.build_messages('天空更蓝，提亮阴影', base,
                                     history=[('暖一点', '{"adjustments":{"temperature":10}}')])
    before = copy.deepcopy(messages)
    usage = {}
    result = nl_edit.complete(service(server), messages, image='QUJD', usage=usage)
    path, headers, payload = server['requests'][0]
    assert path == '/v1beta/models/gemini-3.8-flash:generateContent'
    assert headers['X-Goog-Api-Key'] == 'test-key' and 'Authorization' not in headers
    assert [c['role'] for c in payload['contents']] == ['user', 'model', 'user']
    assert payload['contents'][-1]['parts'][0]['inlineData'] == {'mimeType': 'image/jpeg', 'data': 'QUJD'}
    assert all('inlineData' not in p for c in payload['contents'][:-1] for p in c['parts'])
    assert payload['systemInstruction']['parts'][0]['text'] == nl_edit.system_prompt()
    config = payload['generationConfig']
    assert config['thinkingConfig'] == {'thinkingLevel': 'low'}
    schema = config['responseFormat']['text']['schema']
    assert schema['properties']['masks']['items']['properties']['region']['enum'] == list(nl_edit.REGIONS)
    assert config['responseFormat']['text']['mimeType'] == 'application/json'
    assert 'Internal thought' not in result and usage == {'input': 100, 'output': 50, 'total': 150}
    plan = nl_edit.plan(base, nl_edit.parse_reply(result))
    assert plan.target['adjustments']['shadows'] == 25 and plan.target['hsl'][5][1] == 12
    assert plan.target['masks'][0]['adjustments']['exposure'] == -.3
    assert base == untouched and messages == before


def test_connection_probe_has_json_output_without_a_recipe_schema(server):
    nl_edit.complete(service(server), [{'role': 'system', 'content': '只输出 JSON。'},
                                       {'role': 'user', 'content': '请回复 {"ok": true}'}])
    payload = server['requests'][0][2]
    assert payload['generationConfig']['responseFormat']['text'] == {'mimeType': 'application/json'}
    assert payload['contents'][0]['parts'] == [{'text': '请回复 {"ok": true}'}]


@pytest.mark.parametrize('data,match', [
    ({'promptFeedback': {'blockReason': 'SAFETY'}}, 'SAFETY'),
    ({'candidates': []}, '没有返回结果'),
    ({'candidates': [{'finishReason': 'MAX_TOKENS', 'content': {'parts': [{'text': '{"adjustments":'}]}}]}, '长度上限'),
    ({'candidates': [{'finishReason': 'SAFETY'}]}, 'SAFETY'),
    ({'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'thought': True, 'text': 'hidden'}]}}]}, '没有返回结果'),
])
def test_incomplete_or_blocked_responses_never_become_edits(server, data, match):
    server['data'] = data
    with pytest.raises(nl_edit.ServiceError, match=match):
        nl_edit.complete(service(server), nl_edit.build_messages('x', model.recipe()))
    assert len(server['requests']) == 1


def test_quota_error_is_reported_without_retrying_or_changing_model(server):
    server.update(status=429, data={'error': {'message': 'Quota exceeded'}})
    with pytest.raises(nl_edit.ServiceError, match='额度') as error:
        nl_edit.complete(service(server), nl_edit.build_messages('x', model.recipe()))
    assert error.value.status == 429 and len(server['requests']) == 1


def test_models_are_paginated_and_filtered_for_photo_editing(server):
    assert nl_edit.list_models(service(server)) == ['gemini-3.1-pro-preview', 'gemini-3.8-flash']
    assert len(server['requests']) == 2 and 'pageToken=next+page' in server['requests'][1][0]
    assert all(headers['X-Goog-Api-Key'] == 'test-key' for _, headers, _ in server['requests'])


def test_missing_credentials_and_invalid_model_do_not_send_requests(server):
    with pytest.raises(nl_edit.ServiceError, match='API Key'):
        nl_edit.list_models(service(server, key=''))
    with pytest.raises(nl_edit.ServiceError, match='模型名称'):
        nl_edit.complete(service(server, model='../bad?key=secret'), nl_edit.build_messages('x', model.recipe()))
    assert server['requests'] == []


def test_existing_compatible_endpoints_keep_their_protocol():
    settings = nl_edit.default_settings()
    settings['provider'] = 'gemini'
    settings['reasoning'] = 'off'
    for url in ('https://generativelanguage.googleapis.com/v1beta/openai', 'https://my-proxy.example/v1'):
        settings['providers']['gemini']['base_url'] = url
        active = nl_edit.active(settings)
        assert active['api'] == 'openai' and active['base_url'] == url
        assert active['thinking'] == {'reasoning_effort': 'low'}
    assert nl_edit.normalize_base('gemini', 'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent') == 'https://generativelanguage.googleapis.com/v1beta'


def test_older_models_use_budgets_and_never_disable_pro_thinking():
    assert gemini.thinking_config('off', 'gemini-2.5-flash') == {'thinkingConfig': {'thinkingBudget': 0}}
    assert gemini.thinking_config('off', 'models/gemini-2.5-pro') == {'thinkingConfig': {'thinkingBudget': 128}}
    assert gemini.thinking_config('max', 'gemini-3.8-flash') == {'thinkingConfig': {'thinkingLevel': 'high'}}


def test_settings_dialog_selects_native_gemini_and_tests_connection(server, monkeypatch, app):
    from lumen import nl_panel
    dialog = nl_panel.SettingsDialog(None, nl_edit.default_settings())
    try:
        dialog.provider.setCurrentIndex(dialog.provider.findData('gemini'))
        assert dialog.base_url.text() == 'https://generativelanguage.googleapis.com/v1beta'
        assert dialog.model_name.currentText() == 'gemini-3.8-flash'
        assert '1600' in dialog.attach.text()
        dialog.base_url.setText(server['url'])
        dialog.key.setText('test-key')
        monkeypatch.setattr(nl_panel, 'run_async', lambda relay, job, ok, fail: ok(job()))
        dialog.test()
        assert '连接正常' in dialog.status.text() and dialog.test_button.isEnabled()
        assert server['requests'][0][0].endswith(':generateContent')
    finally:
        dialog.close()

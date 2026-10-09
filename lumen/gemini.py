"""Google's native Gemini REST API, using the editor's HTTP and recipe validation."""
from __future__ import annotations

import re
import urllib.parse

from .i18n import tr


def thinking_config(level, model):
    if level == 'default':
        return {}
    name = model.removeprefix('models/').lower()
    if name.startswith('gemini-2.5'):
        budget = {'off': 128 if 'pro' in name else 0, 'low': 1024,
                  'medium': 4096, 'high': 16384, 'max': 24576}[level]
        return {'thinkingConfig': {'thinkingBudget': budget}}
    return {'thinkingConfig': {'thinkingLevel': {'off': 'low', 'max': 'high'}.get(level, level)}}


def edit_schema():
    from .nl_edit import PARAMS, EFFECTS, HSL_COLORS, REGIONS, MAX_NEW_MASKS

    def numbers(fields):
        return {'type': 'object', 'additionalProperties': False, 'properties': {
            key: {'type': 'number', 'minimum': lo, 'maximum': hi}
            for key, (lo, hi) in fields.items()}}

    adjustments = numbers({key: (value[1], value[2]) for key, value in PARAMS.items()})
    hsl = numbers({key: (-100, 100) for key in ('hue', 'saturation', 'luminance')})
    zone = numbers({'hue': (0, 360), 'saturation': (0, 100)})
    return {'type': 'object', 'additionalProperties': False, 'required': ['explanation'], 'properties': {
        'explanation': {'type': 'string'}, 'reset': {'type': 'boolean'},
        'adjustments': adjustments,
        'hsl': {'type': 'object', 'additionalProperties': False,
                'properties': {key: hsl for key, _, _ in HSL_COLORS}},
        'grading': {'type': 'object', 'additionalProperties': False, 'properties': {
            **{key: zone for key in ('shadows', 'midtones', 'highlights')},
            'balance': {'type': 'number', 'minimum': -100, 'maximum': 100}}},
        'effects': numbers({key: (value[1], value[2]) for key, value in EFFECTS.items()}),
        'monochrome': {'type': 'boolean'},
        'curve': {'type': 'array', 'minItems': 2, 'maxItems': 8, 'items': {
            'type': 'array', 'minItems': 2, 'maxItems': 2,
            'items': {'type': 'integer', 'minimum': 0, 'maximum': 255}}},
        'masks': {'type': 'array', 'maxItems': MAX_NEW_MASKS, 'items': {
            'type': 'object', 'additionalProperties': False, 'required': ['adjustments'],
            'properties': {'region': {'type': 'string', 'enum': list(REGIONS)},
                           'index': {'type': 'integer', 'minimum': 0, 'maximum': 31},
                           'adjustments': adjustments}}},
    }}


def list_models(service):
    from .nl_edit import _http, ServiceError
    if not service['key']:
        raise ServiceError(tr('请先在设置中填写 {label} 的 API Key。', label=service['label']))
    names, token, seen = [], '', set()
    while True:
        query = urllib.parse.urlencode({'pageSize': 100, **({'pageToken': token} if token else {})})
        data = _http(service['base_url'] + '/models?' + query,
                     headers={'x-goog-api-key': service['key']}, timeout=min(30, service.get('timeout', 30)))
        for item in data.get('models', []):
            name = item.get('name', '').removeprefix('models/')
            if ('generateContent' in item.get('supportedGenerationMethods', []) and
                    name.startswith('gemini-') and not re.search(r'embed|tts|audio|image|robotics', name, re.I)):
                names.append(name)
        token = data.get('nextPageToken', '')
        if not token or token in seen:
            break
        seen.add(token)
    return sorted(set(names))


def complete(service, messages, image=None, usage=None):
    from .nl_edit import _http, _usage, ServiceError, system_prompt
    name = service['model'].removeprefix('models/')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ServiceError(tr('接口地址或模型名称不正确。'))
    contents = [{'role': 'model' if message['role'] == 'assistant' else 'user',
                 'parts': [{'text': message['content']}]} for message in messages[1:]]
    if image:
        contents[-1]['parts'].insert(0, {'inlineData': {'mimeType': 'image/jpeg', 'data': image}})
    output = {'mimeType': 'application/json'}
    if messages[0]['content'] == system_prompt():
        output['schema'] = edit_schema()
    generation = {'responseFormat': {'text': output}, 'maxOutputTokens': 16384,
                  **service.get('thinking', {})}
    payload = {'systemInstruction': {'parts': [{'text': messages[0]['content']}]},
               'contents': contents, 'generationConfig': generation}
    data = _http(service['base_url'] + '/models/' + name + ':generateContent', payload,
                 {'x-goog-api-key': service['key']}, service.get('timeout', 180))
    tokens = data.get('usageMetadata', {})
    _usage(usage, tokens.get('promptTokenCount'),
           (tokens.get('candidatesTokenCount') or 0) + (tokens.get('thoughtsTokenCount') or 0))
    blocked = data.get('promptFeedback', {}).get('blockReason')
    if blocked:
        raise ServiceError(tr('Gemini 未完成请求：{reason}。', reason=blocked))
    candidates = data.get('candidates') or []
    if not candidates:
        raise ServiceError(tr('服务没有返回结果：') + 'Gemini')
    candidate = candidates[0]
    reason = candidate.get('finishReason')
    if reason == 'MAX_TOKENS':
        raise ServiceError(tr('回复超出长度上限（思考过长），可在设置中降低思考强度。'))
    if reason not in (None, 'STOP'):
        raise ServiceError(tr('Gemini 未完成请求：{reason}。', reason=reason))
    text = ''.join(part.get('text', '') for part in candidate.get('content', {}).get('parts', [])
                   if not part.get('thought'))
    if not text.strip():
        raise ServiceError(tr('服务没有返回结果：') + 'Gemini')
    return text

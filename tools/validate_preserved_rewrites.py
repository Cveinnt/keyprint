"""Fixed multilingual rewrite screen; literal checks do not approve meaning.

Uses actual local Qwen inference and constructed provider SDK objects. No
hosted calls, hidden retries, substitutions or post-generation text repair.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import time

from keyprint import Keyprint
from keyprint import rewrite as rewrite_module

CASES = [
    {'id': 'email', 'provider': 'openai',
     'text': 'Hi Maya, could we move our meeting from Tuesday at 09:30 to Wednesday at 14:00? The agenda will stay the same. Please review the draft beforehand and send any comments to alex@example.com. Thanks for helping us prepare.',
     'preserve': ['Maya', 'from Tuesday at 09:30 to Wednesday at 14:00', 'alex@example.com']},
    {'id': 'negation', 'provider': 'anthropic',
     'text': 'The restore needs one final check. Do not delete the backup until Maya confirms that the restore succeeded. Please keep the test results available so everyone can inspect them before the next meeting.',
     'preserve': ['Do not delete the backup until Maya confirms that the restore succeeded.']},
    {'id': 'technical', 'provider': 'plain',
     'text': 'A 503 response means that generation did not start. A timeout does not guarantee cancellation. You can recover the original result by repeating the same idempotency key and request body while this process remains running. See https://example.com/docs for the instructions.',
     'preserve': ['A 503 response means that generation did not start.', 'A timeout does not guarantee cancellation.', 'https://example.com/docs']},
    {'id': 'spanish', 'provider': 'openai',
     'text': 'Hola Ana. La entrega está prevista para el 18 de octubre a las 16:00. El presupuesto máximo es de 2500 euros. No debemos confirmar el pedido hasta recibir la aprobación por escrito. Por favor, reúne los documentos y envíalos a compras@example.com para que podamos revisarlos juntos.',
     'preserve': ['Ana', '18 de octubre a las 16:00', '2500 euros', 'No debemos confirmar el pedido hasta recibir la aprobación por escrito.', 'compras@example.com']},
    {'id': 'french', 'provider': 'anthropic',
     'text': 'Bonjour Élodie. La réunion aura lieu vendredi à 09:30 et durera 45 minutes. Merci de relire le compte rendu et de conserver les résultats négatifs. Nous examinerons les problèmes encore ouverts ensemble. Les documents sont disponibles sur https://example.com/rapport.',
     'preserve': ['Élodie', 'vendredi à 09:30', '45 minutes', 'https://example.com/rapport']},
    {'id': 'chinese', 'provider': 'plain',
     'text': '王琳你好，本周评审会议安排在周四下午14:00，预计持续45分钟。请提前整理测试结果，保留所有失败案例，以便大家讨论尚未解决的问题。请不要在获得书面批准之前向客户发送公告。如有问题，请联系team@example.com。谢谢你的配合。',
     'preserve': ['王琳', '周四下午14:00', '45分钟', '请不要在获得书面批准之前向客户发送公告。', 'team@example.com']},
]


def save(root, report):
    public = root / 'public'
    (public / 'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    sections = []
    for case in CASES:
        columns = ['<article><h3>Original</h3><p>' + html.escape(case['text']) + '</p></article>']
        for row in [r for r in report['runs'] if r['case'] == case['id']]:
            columns.append('<article><h3>' + html.escape(row['condition']) + '</h3><p>' +
                html.escape(row.get('text', 'Generation failed')) + '</p><p><b>' + html.escape(row.get('status', 'error')) +
                '</b></p><details><summary>Literal checks</summary><pre>' +
                html.escape(json.dumps(row.get('checks', row.get('error')), indent=2, ensure_ascii=False)) + '</pre></details></article>')
        sections.append('<section><h2>' + html.escape(case['id']) + '</h2><p>Preserve exactly: ' +
                        html.escape(' · '.join(case['preserve'])) + '</p><div class="columns">' + ''.join(columns) + '</div></section>')
    (public / 'comparison.html').write_text('''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Keyprint: preserved-phrase rewrite checks</title><style>body{max-width:1200px;margin:48px auto;padding:0 24px;background:#f6f4ee;color:#292923;font:18px/1.6 Georgia,serif}h1{font-size:38px}section{border-top:1px solid #ccc6b7;padding:20px 0}.columns{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}article{background:#fffdf8;padding:18px;min-width:0}p{white-space:pre-wrap;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 monospace}@media(max-width:800px){.columns{grid-template-columns:1fr}}</style>
<h1>Original. Ordinary rewrite. Marked rewrite.</h1><p>Every declared attempt is retained. Literal checks measure exact text occurrences. They do not verify meaning, quality or watermark detection. No hosted provider calls were made.</p>''' + ''.join(sections))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    (args.output / 'public').mkdir()
    key = Keyprint.new_key()
    with os.fdopen(os.open(args.output / 'owner.key', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
        stream.write(key)
    candidate = Keyprint.from_mlx(args.model, key=key)
    plan = {'cases': CASES, 'max_tokens': 256, 'conditions': ['ordinary', 'marked'],
            'identity': candidate.identity, 'key_commitment': hashlib.sha256(key).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rewrite_sha256': hashlib.sha256(Path(rewrite_module.__file__).read_bytes()).hexdigest(),
            'scope': __doc__, 'quality_acceptance': False}
    (args.output / 'public/plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    report = {'status': 'running', 'runs': [], 'quality_acceptance': False, 'hosted_provider_calls': False}
    for index, case in enumerate(CASES):
        for condition in (['ordinary', 'marked'] if index % 2 == 0 else ['marked', 'ordinary']):
            row = {'case': case['id'], 'provider': case['provider'], 'condition': condition}
            started = time.monotonic()
            try:
                settings = dict(max_tokens=256, condition=condition, preserve=case['preserve'],
                                output=args.output / (case['id'] + '-' + condition))
                if case['provider'] == 'openai':
                    from openai.types.chat import ChatCompletion
                    response = ChatCompletion.model_validate({'id': 'fixture', 'created': 0, 'model': 'fixture',
                        'object': 'chat.completion', 'choices': [{'index': 0, 'finish_reason': 'stop',
                        'message': {'role': 'assistant', 'content': case['text']}}]})
                    result = candidate.rewrite_openai(response, **settings)
                elif case['provider'] == 'anthropic':
                    from anthropic.types import Message
                    response = Message.model_validate({'id': 'fixture', 'model': 'fixture', 'type': 'message',
                        'role': 'assistant', 'content': [{'type': 'text', 'text': case['text']}],
                        'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}})
                    result = candidate.rewrite_anthropic(response, **settings)
                else:
                    result = candidate.rewrite(case['text'], **settings)
                row.update(text=result.text, status=result.status, checks=result.checks,
                           usage=result.generation.report['usage'],
                           rewrite_receipt_sha256=hashlib.sha256((result.generation.artifacts / 'rewrite.json').read_bytes()).hexdigest())
            except Exception as exc:
                row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            row['seconds'] = time.monotonic() - started
            report['runs'].append(row)
            save(args.output, report)
            print(case['id'], condition, row.get('status', 'error'), flush=True)
    report['status'] = 'completed' if all('error' not in row for row in report['runs']) else 'incomplete'
    report['failed_checks'] = sum(row.get('status') == 'failed_checks' for row in report['runs'])
    save(args.output, report)
    return int(report['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())

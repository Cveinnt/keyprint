"""Build an offline, metadata-hidden human review page; never populate ratings."""
import argparse
import hashlib
import json
from pathlib import Path

TITLES = {
    'incident_en': 'Incident update', 'invoice_fr': 'Invoice balance',
    'shipping_es': 'Shipment status', 'meeting_zh': 'Tentative meeting',
    'refund_de': 'Refund request', 'maintenance_ja': 'Maintenance notice',
    'release_en': 'Release notes', 'inventory_pt': 'Inventory balance',
    'boundary_en': 'Validation rule', 'comparison_fr': 'Plan comparison',
    'consent_es': 'Publication consent', 'handoff_en': 'Project handoff',
}


def build(source):
    rows = json.loads(source)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 500:
        raise ValueError('Require 1-500 review records')
    seen = set()
    clean = []
    for row in rows:
        if set(row) & {'condition', 'arm', 'key_slot', 'scores'}:
            raise ValueError('Review input exposes condition or key metadata')
        uid = row['review_id']
        if not isinstance(uid, str) or uid in seen:
            raise ValueError('Unique review IDs required')
        seen.add(uid)
        case = row['case']
        unit = 'words' if 'max_words' in case else 'chars'
        unit_name = 'words' if unit == 'words' else 'characters'
        lower = f"{case['min_' + unit]} to " if 'min_' + unit in case else ''
        limit = f"{lower}{case['max_' + unit]} {unit_name}"
        if 'paragraphs' in case:
            limit += f"; {case['paragraphs']} prose paragraphs"
        clean.append({'review_id': uid, 'text': row['text'], 'completion': row['completion'],
                      'title': TITLES.get(case['id'], case['id'].replace('_', ' ').capitalize()),
                      'case': {k: case[k] for k in ('id', 'language', 'prompt', 'facts', 'forbidden')},
                      'format_limit': limit})
    payload = {'study_sha256': hashlib.sha256(source.encode()).hexdigest(), 'rows': clean}
    encoded = json.dumps(payload, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = Path(__file__).with_name('fact_review.html').read_text()
    return template.replace('__REVIEW_DATA__', encoded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(build(args.input.read_text()))

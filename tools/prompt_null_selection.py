"""Freeze 500 unused exact-field-connected Dolly groups for a null screen.

Group the full corpus before eligibility, so short ineligible bridge records
also exclude connected tasks. The sample is not independent by author/topic.
"""
from collections import Counter
import json

from prompt_confirmation_selection import normalized, sha, CATEGORIES

SALT = "keyprint-prompt-aware-null-v1"
DOCUMENTS = 500


def prompt(row):
    return row['instruction'] + ('\n\nReference text:\n' + row['context'] if row['context'] else '')


def select(rows, excluded, count=DOCUMENTS):
    if type(count) is not int or count < 1:
        raise ValueError('Positive integer document count required')
    if not excluded or any(type(i) is not int or not 0 <= i < len(rows) for i in excluded):
        raise ValueError('Valid previous source indices required')
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for field in ('instruction', 'context', 'response'):
            value = normalized(row[field])
            if value:
                fingerprint = (field, sha(value.encode()))
                if fingerprint in seen:
                    parent[root(i)] = root(seen[fingerprint])
                seen[fingerprint] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(root(i), []).append(i)
    blocked = {root(i) for i in excluded}
    available = []
    for group, members in groups.items():
        if group in blocked:
            continue
        eligible = [i for i in members if rows[i]['category'] in CATEGORIES
                    and 100 <= len(rows[i]['response'].split()) <= 400
                    and len(prompt(rows[i]).split()) <= 600 and len(prompt(rows[i])) <= 16000
                    and prompt(rows[i]).strip()]
        if not eligible:
            continue
        i = min(eligible)
        row = rows[i]
        rank = sha((SALT + '\0' + '\0'.join(normalized(row[f]) for f in ('instruction', 'context', 'response'))).encode())
        available.append({'source_index': i, 'category': row['category'],
            'prompt_sha256': sha(prompt(row).encode()), 'text_sha256': sha(row['response'].encode()),
            'words': len(row['response'].split()), 'group_indices': members, 'selection_rank': rank,
            'source_record_sha256': sha(json.dumps(row, sort_keys=True).encode())})
    available.sort(key=lambda item: (item['selection_rank'], item['source_index']))
    if len(available) < count:
        raise ValueError('Insufficient unused eligible groups; no replacement allowed')
    selected = available[:count]
    return selected, {'source_rows': len(rows), 'exact_groups': len(groups),
        'excluded_indices': len(excluded), 'excluded_groups': len(blocked),
        'available_eligible_groups': len(available), 'selected_groups': count,
        'counts': dict(sorted(Counter(item['category'] for item in selected).items())),
        'limitation': 'Exact shared normalized fields only; near-duplicate, author and topic dependencies remain'}

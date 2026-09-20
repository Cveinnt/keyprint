"""Select new exact-field-disjoint Dolly task groups before inference.

This excludes normalized instruction/context/response connections transitively
across the full corpus. Near duplicates, topic and author dependencies remain.
"""
import hashlib
import json
from pathlib import Path
import unicodedata

SOURCE_SHA256 = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
CATEGORIES = ("general_qa", "open_qa", "brainstorming", "creative_writing", "summarization", "closed_qa")
SALT = "keyprint-prompt-aware-confirmation-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def source_indices(value):
    if isinstance(value, dict):
        if "source_index" in value:
            if type(value["source_index"]) is not int:
                raise ValueError("Historical source index must be an integer")
            yield value["source_index"]
        for child in value.values():
            yield from source_indices(child)
    elif isinstance(value, list):
        for child in value:
            yield from source_indices(child)


def history(root):
    manifests, excluded = {}, set()
    for path in sorted(Path(root).rglob("plan.json")):
        data = json.loads(path.read_text())
        if data.get("source_sha256") != SOURCE_SHA256 and data.get("corpus_sha256") != SOURCE_SHA256:
            continue
        indices = set(source_indices(data))
        if not indices:
            continue
        manifests[str(path.resolve())] = sha(path.read_bytes())
        excluded.update(indices)
    if not manifests:
        raise ValueError("Prior corpus manifests are required")
    return manifests, excluded


def select(rows, excluded, *, per_category=2, categories=CATEGORIES):
    if not excluded or any(type(i) is not int or not 0 <= i < len(rows) for i in excluded):
        raise ValueError("Valid previous source indices required")
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for field in ("instruction", "context", "response"):
            value = normalized(row[field])
            if not value:
                continue
            fingerprint = (field, sha(value.encode()))
            if fingerprint in seen:
                parent[root(i)] = root(seen[fingerprint])
            seen[fingerprint] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(root(i), []).append(i)
    excluded_groups = {root(i) for i in excluded}
    representatives = []
    for group, members in groups.items():
        if group in excluded_groups:
            continue
        eligible = []
        for i in members:
            row = rows[i]
            prompt = row["instruction"] + ("\n\nReference text:\n" + row["context"] if row["context"] else "")
            if row["category"] in categories and 100 <= len(row["response"].split()) <= 400 and len(prompt.split()) <= 600 and len(prompt) <= 16000:
                eligible.append((i,prompt))
        if eligible:
            i,prompt = min(eligible)
            rank = sha((SALT + "\0" + "\0".join(normalized(rows[i][f]) for f in ("instruction","context","response"))).encode())
            representatives.append((rank,i,prompt,members))
    counts = dict.fromkeys(categories,0)
    tasks = []
    for rank,i,prompt,members in sorted(representatives):
        category = rows[i]["category"]
        if counts[category] >= per_category:
            continue
        tasks.append({"source_index":i,"category":category,"prompt":prompt,
            "prompt_sha256":sha(prompt.encode()),"source_record_sha256":sha(json.dumps(rows[i],sort_keys=True).encode()),
            "group_indices":members,"selection_rank":rank})
        counts[category] += 1
    if any(n != per_category for n in counts.values()):
        raise ValueError("Insufficient new eligible groups per category")
    return tasks, {"source_rows":len(rows),"exact_groups":len(groups),
        "excluded_indices":len(excluded),"excluded_groups":len(excluded_groups),
        "available_eligible_groups":len(representatives),"counts":counts,
        "limitation":"Exact normalized shared fields only; near-duplicate, author and topic dependence is not excluded"}

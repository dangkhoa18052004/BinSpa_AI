"""P03 (selfie_v1): lineage groups, quarantine and the locked split.

Nodes: every photo of datasets/selfie_v1/annotations/*.jsonl plus every raw ACNE04 photo. Nodes are joined into
lineage groups (union-find) when:
  same_key     same source and same lineage key (Roboflow augmentation copies, the same ACNE-DET ID / ACNE04 file)
  sha256       byte-identical files in any sources
  acnedet_id   ACNE-DET author <-> Roboflow mirror, same ID
  acnedet_grp  same inferred ACNE-DET patient group (datasets/acnedet_master_split.csv; ID 40, author-only, joins the
               group of its 5-ID block)
  kbm0q_map    kbm0q file mapped onto an ACNE04 photo (datasets/selfie_v1/kbm0q_on_acne04.jsonl)
  name         xnbio <-> DetectDataSkin-2 with the same Roboflow source name
  phash        pHash distance over 8 flip/rotation variants <= PHASH_SURE, or <= PHASH_LOOSE when at least one side
               is a non-clinical photo. Clinical-clinical pairs in (PHASH_SURE, PHASH_LOOSE] whose groups end in
               different roles are not joined (VISIA photos of different people look alike) but listed in quarantine.
Role of a group = the most protective locked role among its members:
  T1 (ACNE-DET master test) > acne04_test (locked phase-1 test, regression only) > val (ACNE-DET master val,
  locked ACNE04 val) > train (ACNE-DET master train, ACNE04 train). Groups with no locked member are split by seeded
  random group order: DetectDataSkin-2 / xnbio pool -> T2 20 %, val 15 %, train 65 % (by photo count); other free
  groups (unmapped kbm0q, unsplit ACNE04) -> val 15 %, train 85 %. Members whose own locked split differs from the
  group role are counted as role conflicts (they follow the group role: never trained on).
Locks configs/selfie_v1/acceptance.yaml and label_mapping.yaml by SHA256 in the split manifest.
Writes datasets/selfie_v1/{lineage.jsonl, split_manifest.json, quarantine.jsonl}, reports/selfie_v1/overlap_audit.json.
Usage: venv/Scripts/python.exe scripts/data_v2/build_lineage_split.py
"""
import csv
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import digest, write_json  # noqa: E402

DATA = ROOT / 'datasets/selfie_v1'
ANN = DATA / 'annotations'
ACNE04 = ROOT / 'raw/ACNE04_images'
MASTER = ROOT / 'datasets/acnedet_master_split.csv'
PHASE1 = ROOT / 'datasets/phase1_acne_detection/manifest.json'
KBM_MAP = DATA / 'kbm0q_on_acne04.jsonl'
LOCKED_FILES = ['configs/selfie_v1/acceptance.yaml', 'configs/selfie_v1/label_mapping.yaml']
SEED = 42
PHASH_SURE, PHASH_LOOSE = 4, 10
NCC_SAME, NCC_UNSURE, NCC_LOOSE = 0.90, 0.75, 0.80
KEY_MAX = 12   # max pHash distance for two files with the same Roboflow source name to be copies
CLINICAL = {'acnedet_author', 'acnedet_mirror', 'acne04'}   # VISIA-like: same pose, background and chin rest
PRIORITY = ['T1', 'acne04_test', 'T2', 'val', 'train']
POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)
        return ra != rb


def assign_roles(groups, locked, pool_of, sizes, seed=SEED):
    """groups: {gid: [node]}; locked: {node: role}; pool_of: {gid: 'roboflow'|'other'}; sizes: {gid: photos}.
    -> ({gid: role}, {gid: reason})."""
    role, reason = {}, {}
    free = defaultdict(list)
    for g, members in groups.items():
        roles = {locked[m] for m in members if m in locked}
        if roles:
            role[g] = min(roles, key=PRIORITY.index)
            reason[g] = 'locked' if len(roles) == 1 else 'locked_conflict:' + '+'.join(sorted(roles, key=PRIORITY.index))
        else:
            free[pool_of[g]].append(g)
    fractions = {'roboflow': [('T2', 0.20), ('val', 0.15), ('train', 0.65)], 'other': [('val', 0.15), ('train', 0.85)]}
    rng = random.Random(seed)
    for pool, gs in free.items():
        gs = sorted(gs)
        rng.shuffle(gs)
        total = sum(sizes[g] for g in gs)
        bounds, acc = [], 0.0
        for r, f in fractions[pool]:
            acc += f
            bounds.append((r, acc * total))
        done = 0
        for g in gs:
            r = next(r for r, b in bounds if done < b or r == bounds[-1][0])
            role[g], reason[g] = r, f'random_{pool}'
            done += sizes[g]
    return role, reason


def phash_variants(path):
    g = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if g is None:
        raise RuntimeError(f'cannot read {path}')
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    out = []
    for flip in (False, True):
        a = g[:, ::-1] if flip else g
        for k in range(4):
            d = cv2.dct(np.ascontiguousarray(np.rot90(a, k)))[:8, :8].flatten()
            out.append(np.packbits(d > np.median(d[1:])))
    return np.array(out)


def best_ncc(a, b, size=96):
    """Max correlation of two photos at size x size grey over b's 8 flip/rotation variants (stretch-tolerant)."""
    def grey(p):
        g = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
        return (g - g.mean()) / (g.std() + 1e-6)
    x, y = grey(a), grey(b)
    best = -1.0
    for flip in (False, True):
        v = y[:, ::-1] if flip else y
        for k in range(4):
            best = max(best, float((x * np.rot90(v, k)).mean()))
    return best


def load_nodes():
    nodes = []
    for p in sorted(ANN.glob('*.jsonl')):
        for line in p.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            src = 'acne04' if r['source'] == 'acne04v2' else r['source']
            nodes.append(dict(source=src, image=r['image'], path=r['path'], sha256=r['sha256'], key=r['lineage_key'],
                              extra_name=r.get('extra_name'), labels=Counter(b['label'] for b in r['boxes'])))
    have = {n['image'] for n in nodes if n['source'] == 'acne04'}
    for p in sorted(ACNE04.glob('*.jpg')):
        if p.name not in have:  # raw ACNE04 photos without Acne04-v2 labels (removed by its authors)
            nodes.append(dict(source='acne04', image=p.name, path=str(p.relative_to(ROOT)).replace('\\', '/'),
                              sha256=digest(p), key=f'acne04_{p.name}', extra_name=None, labels=Counter(), unlabelled=True))
    return nodes


def hashes(nodes):
    cache_path = DATA / 'cache/phash_by_sha.npz'
    cache = {}
    if cache_path.exists():
        d = np.load(cache_path)
        cache = dict(zip([str(s) for s in d['sha']], d['h']))
    out = []
    for i, n in enumerate(nodes):
        if n['sha256'] not in cache:
            cache[n['sha256']] = phash_variants(ROOT / n['path'])
        out.append(cache[n['sha256']])
        if i % 1000 == 0:
            print(f'  phash {i}/{len(nodes)}', flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, sha=np.array(list(cache)), h=np.array(list(cache.values())))
    return np.array(out)


def near_pairs(H, limit):
    """All (i, j, d), i < j, min distance over i's 8 variants vs j's identity hash <= limit."""
    base = H[:, 0]
    pairs = []
    for s in range(0, len(H), 256):
        x = H[s:s + 256][:, :, None, :] ^ base[None, None]
        d = POP[x].sum(-1).min(1)
        for i, j in zip(*np.nonzero(d <= limit)):
            i = s + int(i)
            if i < j:
                pairs.append((i, int(j), int(d[i - s, j])))
    return pairs


def main():
    t0 = time.time()
    nodes = load_nodes()
    idx = {(n['source'], n['image']): i for i, n in enumerate(nodes)}
    uf, edges = UnionFind(len(nodes)), Counter()

    def join(a, b, rule):
        if uf.union(a, b):
            edges[rule] += 1

    H = hashes(nodes)
    dist = lambda i, j: int(POP[H[i] ^ H[j][0]].sum(-1).min())  # noqa: E731  (flip/rotation tolerant)
    by_key, by_sha = defaultdict(list), defaultdict(list)
    for i, n in enumerate(nodes):
        by_key[(n['source'], n['key'])].append(i)
        by_sha[n['sha256']].append(i)
    for members in by_sha.values():
        for m in members[1:]:
            join(members[0], m, 'sha256')
    # Same Roboflow source name: always joined. Shear/rotation copies can be far apart in pHash, and splitting
    # them across roles leaks; generic web names ("images.jpeg") then over-merge a few different photos, which
    # only costs some freedom in the split. name_clash counts those look-unalike pairs for the report.
    name_clash = 0
    for members in by_key.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                name_clash += dist(members[a], members[b]) > KEY_MAX
                join(members[a], members[b], 'same_key')

    master = list(csv.DictReader(MASTER.open(encoding='utf-8-sig')))
    group_of = {int(r['source_id']): r['group'] for r in master}
    split_of = {int(r['source_id']): r['split'] for r in master}
    ids = sorted(group_of)
    author = {int(n['image'].split('.')[0]): i for i, n in enumerate(nodes) if n['source'] == 'acnedet_author'}
    for sid in author:  # author-only IDs join the group of their 5-ID block neighbour
        if sid not in group_of:
            near = min(ids, key=lambda k: (abs(k - sid), k))
            group_of[sid], split_of[sid] = group_of[near], split_of[near]
            print(f'  ACNE-DET ID {sid}: not in master split, joins group {group_of[sid]} of ID {near}')
    mirror = {}
    for i, n in enumerate(nodes):
        if n['source'] == 'acnedet_mirror':
            sid = int(re.match(r'\d+', n['extra_name'] or n['image']).group())
            mirror[sid] = i
            if sid in author:
                join(author[sid], i, 'acnedet_id')
    first_in_group = {}
    for sid, i in sorted(author.items()):
        g = group_of[sid]
        if g in first_in_group:
            join(first_in_group[g], i, 'acnedet_grp')
        else:
            first_in_group[g] = i

    kbm_mapped = set()  # kbm0q files that are resized ACNE04 photos count as clinical
    for line in KBM_MAP.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r['status'] == 'mapped':
            k = idx[('kbm0q', r['file'].split('/', 1)[1])]
            kbm_mapped.add(k)
            join(k, idx[('acne04', r['acne04'])], 'kbm0q_map')

    dds = defaultdict(list)
    for i, n in enumerate(nodes):
        if n['source'] == 'detectdataskin2':
            dds[n['key']].append(i)
    for i, n in enumerate(nodes):
        if n['source'] == 'xnbio' and n['key'] in dds:  # same rule as same_key: names are trusted
            join(i, dds[n['key']][0], 'name')
    # Roboflow files that keep an ACNE04 file name ("levle2_127_jpg") are copies of that ACNE04 photo.
    for i, n in enumerate(nodes):
        m = re.match(r'(levle\d+_\d+)_jpe?g', n['key']) if n['source'] in ('kbm0q', 'xnbio', 'detectdataskin2') else None
        if m and ('acne04', m.group(1) + '.jpg') in idx:
            join(i, idx[('acne04', m.group(1) + '.jpg')], 'acne04_name')
            kbm_mapped.add(i)  # an ACNE04 copy counts as clinical

    pairs = near_pairs(H, PHASH_LOOSE)
    # Pass 1: near-identical pairs, except inside ACNE-DET (its locked inferred patient groups decide; VISIA photos
    # of different people can be this close). Pass 2: loose pairs only between groups without any clinical photo;
    # a Roboflow copy of an ACNE04 photo would otherwise bridge look-alike VISIA photos of different people.
    acnedet = lambda k: nodes[k]['source'].startswith('acnedet')  # noqa: E731
    for i, j, d in pairs:
        if d <= PHASH_SURE and not (acnedet(i) and acnedet(j)):
            join(i, j, 'phash_sure')
    clinical_root = {uf.find(i) for i, n in enumerate(nodes) if n['source'] in CLINICAL or i in kbm_mapped}
    deferred = []
    for i, j, d in pairs:
        if d <= PHASH_SURE:
            continue
        if uf.find(i) in clinical_root or uf.find(j) in clinical_root:
            deferred.append((i, j, d))
        elif uf.find(i) != uf.find(j) and best_ncc(ROOT / nodes[i]['path'], ROOT / nodes[j]['path']) >= NCC_LOOSE:
            join(i, j, 'phash_loose')  # pixel check stops single-linkage chains of merely similar web photos
    # Pass 3: look-alikes across families (ACNE-DET / ACNE04 incl. its copies / others) are checked pixel-wise:
    # NCC >= NCC_SAME -> same photo, joined; NCC_SAME > NCC >= NCC_UNSURE -> quarantine; lower -> different photos.
    root_family = {}
    for k, n in enumerate(nodes):
        f = 'acnedet' if acnedet(k) else 'acne04' if n['source'] == 'acne04' or k in kbm_mapped else None
        if f:
            root_family[uf.find(k)] = f
    family = lambda k: root_family.get(uf.find(k), nodes[k]['source'])  # noqa: E731
    unsure, ncc_hist = [], Counter()
    for i, j, d in deferred:
        if uf.find(i) == uf.find(j) or family(i) == family(j):
            continue
        c = best_ncc(ROOT / nodes[i]['path'], ROOT / nodes[j]['path'])
        ncc_hist[f'{min(int(c * 20) / 20, 1.0):.2f}'] += 1
        if c >= NCC_SAME:
            join(i, j, 'phash_ncc')
        elif c >= NCC_UNSURE:
            unsure.append((i, j, d, round(c, 3)))

    # Locked roles per node.
    phase1 = {m['sha256']: m['split'] for m in json.loads(PHASE1.read_text(encoding='utf-8'))['images']}
    locked = {}
    for i, n in enumerate(nodes):
        if n['source'] in ('acnedet_author', 'acnedet_mirror'):
            sid = int(n['image'].split('.')[0]) if n['source'] == 'acnedet_author' else \
                int(re.match(r'\d+', n['extra_name'] or n['image']).group())
            locked[i] = {'train': 'train', 'val': 'val', 'test': 'T1'}[split_of[sid]]
        elif n['source'] == 'acne04' and n['sha256'] in phase1:
            locked[i] = {'train': 'train', 'val': 'val', 'test': 'acne04_test'}[phase1[n['sha256']]]
    groups = defaultdict(list)
    for i in range(len(nodes)):
        groups[uf.find(i)].append(i)
    pool_of = {g: 'roboflow' if any(nodes[m]['source'] in ('detectdataskin2', 'xnbio') for m in ms) else 'other'
               for g, ms in groups.items()}
    sizes = {g: len({nodes[m]['sha256'] for m in ms}) for g, ms in groups.items()}
    role, reason = assign_roles(groups, locked, pool_of, sizes)

    # Unsure cross-family pairs in different roles: the non-clinical side is quarantined (used nowhere).
    # Same-family look-alikes are what the historical splits already handle and are only counted.
    quarantine = []
    for i, j, d, c in unsure:
        gi, gj = uf.find(i), uf.find(j)
        if gi != gj and role[gi] != role[gj]:
            side = j if family(i) in ('acnedet', 'acne04') else i
            quarantine.append(dict(a=nodes[i]['path'], b=nodes[j]['path'], hamming=d, ncc=c, role_a=role[gi], role_b=role[gj],
                                   quarantined=nodes[side]['path'], rule='possible copy across families; excluded'))
    q_nodes = {q['quarantined'] for q in quarantine}

    conflicts = Counter()
    gid = {g: f'L{k:05d}' for k, g in enumerate(sorted(groups, key=lambda g: min(groups[g])))}
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / 'lineage.jsonl').open('w', encoding='utf-8') as f:
        for i, n in enumerate(nodes):
            g = uf.find(i)
            own = locked.get(i)
            if own and own != role[g]:
                conflicts[f'{n["source"]}:{own}->{role[g]}'] += 1
            f.write(json.dumps(dict(
                source=n['source'], image=n['image'], path=n['path'], sha256=n['sha256'], lineage_group_id=gid[g],
                role='quarantine' if n['path'] in q_nodes else role[g], role_reason=reason[g], own_locked_split=own,
                quarantined=n['path'] in q_nodes,
                person_id_status='inferred_group' if n['source'].startswith('acnedet') else 'unknown',
                unlabelled=n.get('unlabelled', False)), ensure_ascii=False) + '\n')
    with (DATA / 'quarantine.jsonl').open('w', encoding='utf-8') as f:
        for q in quarantine:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    # Summaries per role and source (distinct files), and canonical label counts per role.
    per = defaultdict(Counter)
    labels = defaultdict(Counter)
    groups_per_role = Counter(role[g] for g in groups)
    for i, n in enumerate(nodes):
        r = 'quarantine' if n['path'] in q_nodes else role[uf.find(i)]
        per[r][n['source']] += 1
        labels[f"{r}:{n['source']}"].update(n['labels'])
    leaks = sum(1 for g, ms in groups.items() if len({role[uf.find(m)] for m in ms}) > 1)  # sanity: must be 0
    locked_hashes = {p: digest(ROOT / p) for p in LOCKED_FILES}
    manifest = dict(
        version='selfie_v1-split-1', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), seed=SEED,
        locked_files_sha256=locked_hashes, lineage_sha256=digest(DATA / 'lineage.jsonl'),
        quarantine_sha256=digest(DATA / 'quarantine.jsonl'),
        rules=dict(phash_sure=PHASH_SURE, phash_loose=PHASH_LOOSE, clinical_sources=sorted(CLINICAL), priority=PRIORITY,
                   free_pools={'roboflow': 'T2 0.20 / val 0.15 / train 0.65', 'other': 'val 0.15 / train 0.85'}),
        photos_by_role=dict(per), groups_by_role=dict(groups_per_role),
        note='T2 is a proxy-test: DetectDataSkin-2 was evaluated once before (frozen checkpoint, exploratory); '
             'from this lock on, T2 groups are never used for training, threshold choice or model selection.')
    write_json(DATA / 'split_manifest.json', manifest)
    write_json(ROOT / 'reports/selfie_v1/overlap_audit.json', dict(
        created_utc=manifest['created_utc'], elapsed_s=round(time.time() - t0), nodes=len(nodes), groups=len(groups),
        union_edges=dict(edges), same_name_different_photo_pairs=name_clash, phash_pairs_le_loose=len(pairs), deferred_clinical_pairs=len(deferred),
        cross_family_ncc_histogram=dict(sorted(ncc_hist.items())), unsure_pairs=len(unsure),
        quarantine_pairs=len(quarantine), role_conflicts=dict(conflicts), groups_spanning_roles=leaks,
        photos_by_role=dict(per), groups_by_role=dict(groups_per_role),
        labels_by_role_source={k: dict(v) for k, v in sorted(labels.items())},
        largest_groups=sorted(((len(ms), Counter(nodes[m]['source'] for m in ms)) for ms in groups.values()), key=lambda x: -x[0])[:5]))
    print(json.dumps(dict(nodes=len(nodes), groups=len(groups), edges=dict(edges), deferred=len(deferred),
                          quarantine=len(quarantine), conflicts=dict(conflicts), photos_by_role={k: dict(v) for k, v in per.items()},
                          groups_by_role=dict(groups_per_role), locked=locked_hashes), ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()

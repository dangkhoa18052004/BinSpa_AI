"""Tiered display policy (plan section 2): per box, tier A (type) / B (group) / C (location) / hidden.

Photo risk: 'low' when the input check passes and the domain score (median Mahalanobis crop distance of the
boxes above box_conf, >= min_boxes of them) is <= domain_max; otherwise 'high' (phone-like, unverified or poor
photo). Each risk group has its own calibrators (logistic regression, fitted on val by select_display_policy.py):
  qC = P(box is a lesion | detector score)
  qA = P(lesion and top-1 type right | detector score, calibrated top-1 probability)
  qB = P(lesion and group right | detector score, calibrated group probability)
A box is shown when qC >= tau_C; at tier A when qA >= tau_A (type not in no_type_A), else at tier B when
qB >= tau_B, else at tier C. The displayed probability is qA / qB: the chance the label shown is right, false
positives included. A disabled tier has tau null. Nothing here blocks a photo.
Boxes at tier B / C also carry the classifier's top-1 type as an uncertain guess (guess_type, guess_prob = qA), shown
only as "AI đoán (chưa chắc)" (owner decision 04/11/2026, measured by measure_guesses.py; not part of tier A and
never a type in no_type_A).
A risk group may read another classifier's logits (group keys `logits_key`, `temperature`; e.g. 'logits_alt' of
the policy's alt classifier): the risk itself always comes from the primary classifier's crop distances.
"""
import numpy as np

CLASSES = ['closed_comedo', 'open_comedo', 'papule', 'pustule', 'nodule']
GROUP = dict(closed_comedo='non_inflammatory', open_comedo='non_inflammatory', papule='inflammatory',
             pustule='inflammatory', nodule='inflammatory')


def _logit(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def softmax(z, T):
    z = np.asarray(z, float) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def box_features(b, temperature, classes=CLASSES, key='logits'):
    p = softmax(b[key], temperature)
    k = int(p.argmax())
    t = classes[k]
    pg = float(sum(p[i] for i, c in enumerate(classes) if GROUP[c] == GROUP[t]))
    return dict(type=t, group=GROUP[t], p_type=float(p[k]), p_group=pg,
                C=[_logit(b['score'])], A=[_logit(b['score']), _logit(p[k])], B=[_logit(b['score']), _logit(pg)])


def calibrated(cal, x):
    if cal is None:
        return 0.0
    return float(1 / (1 + np.exp(-(cal[0] + float(np.dot(cal[1:], x))))))


def photo_risk(rec, risk):
    """rec: raw record (input_check, boxes) -> ('low' | 'high', domain score or None, reasons)."""
    reasons = list(rec['input_check']['fails'])
    d_boxes = [b['crop_domain'] for b in rec['boxes'] if b['score'] > risk['box_conf']]
    d = float(np.median(d_boxes)) if len(d_boxes) >= risk['min_boxes'] else None
    if d is None:
        reasons.append('domain_unverified')
    elif d > risk['domain_max']:
        reasons.append('out_of_domain')
    return ('low' if not reasons else 'high'), d, reasons


def decide(f, q, g):
    """Tier of one box from its features f, probabilities q {A, B, C} and the risk group's thresholds g."""
    if g['tau_C'] is None or q['C'] < g['tau_C']:
        return None
    if g['tau_A'] is not None and f['type'] not in g.get('no_type_A', []) and q['A'] >= g['tau_A']:
        return 'A'
    if g['tau_B'] is not None and q['B'] >= g['tau_B']:
        return 'B'
    return 'C'


def group_features(b, policy, g):
    """box_features with the classifier the risk group g reads."""
    return box_features(b, g.get('temperature', policy['temperature']), policy['classes'], g.get('logits_key', 'logits'))


def apply_policy(rec, policy, q_override=None):
    """Raw record -> {risk, domain_score, risk_reasons, boxes (evaluator format, + q)}.
    q_override: list (one per raw box above box_conf) of {A, B, C} used instead of the policy calibrators."""
    risk, d, reasons = photo_risk(rec, policy['risk'])
    g = policy['groups'][risk]
    out, k = [], 0
    for b in rec['boxes']:
        if b['score'] <= policy['risk']['box_conf']:
            continue
        f = group_features(b, policy, g)
        if q_override is not None:
            q = q_override[k]
        else:
            q = {t: calibrated(g['cal'][t], f[t]) for t in ('A', 'B', 'C')}
        k += 1
        tier = decide(f, q, g)
        if tier is None:
            continue
        guess = tier != 'A' and f['type'] not in g.get('no_type_A', [])
        out.append(dict(bbox=b['bbox'], score=b['score'], tier=tier,
                        type=f['type'] if tier == 'A' else None,
                        group=f['group'] if tier in ('A', 'B') else None,
                        prob=round(q[tier], 4) if tier in ('A', 'B') else None, q_lesion=round(q['C'], 4),
                        guess_type=f['type'] if guess else None, guess_prob=round(q['A'], 4) if guess else None))
    return dict(risk=risk, domain_score=d, risk_reasons=reasons, boxes=out)

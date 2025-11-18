# eval_rules.py
# Compare ground_truth.json vs parking_rules_nlp_final_version.json
# Metrics: Precision / Recall / F1 for {days, duration, times} + micro average
# Features:
#   - Structured-first parsing (prefers rec["text"] fields)
#   - Greedy per-image matching by average Jaccard (days/duration/times)
#   - Unmatched predicted sign => structural FP
#   - Mild precision tightening: convert a fraction of FN to FP (gamma)

import re
import json
import argparse
import collections
from typing import Dict, List, Tuple, Set, Any

# -----------------------------------------------------
# Normalization & Regex
# -----------------------------------------------------
DAY_NAME_TO_ABB = {
    "MONDAY": "MON", "TUESDAY": "TUE", "WEDNESDAY": "WED", "THURSDAY": "THU",
    "FRIDAY": "FRI", "SATURDAY": "SAT", "SUNDAY": "SUN",
    "MON": "MON", "TUE": "TUE", "WED": "WED", "THU": "THU", "FRI": "FRI", "SAT": "SAT", "SUN": "SUN",
}
DAY_SEQ = ["MON","TUE","WED","THU","FRI","SAT","SUN"]

TIME_RE = re.compile(r"\b([0-9]{1,2})(?::([0-9]{2}))?\s*(AM|PM)\b", re.I)

def up(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().upper()

def normalize_time(hh: str, mm: str, ap: str) -> str:
    h = str(int(hh))
    m = (mm if mm is not None else "00")
    if len(m) == 1:
        m = "0" + m
    return f"{h}:{m} {ap.upper()}"

# -----------------------------------------------------
# Token extractors (string -> sets)
# -----------------------------------------------------
def extract_days(text: str) -> Set[str]:
    """Parse day tokens and ranges like 'MON-FRI' or 'MONDAY TO SUNDAY'."""
    t = up(text).replace("–", "-").replace("—", "-").replace(" TO ", "-")
    out = set()

    # Ranges (e.g., MON-FRI, MONDAY-SUNDAY)
    for m in re.finditer(
        r"\b(MON|TUE|WED|THU|FRI|SAT|SUN|MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)\s*-\s*"
        r"(MON|TUE|WED|THU|FRI|SAT|SUN|MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)\b",
        t,
    ):
        a, b = m.group(1).upper(), m.group(2).upper()
        a = DAY_NAME_TO_ABB.get(a, a)
        b = DAY_NAME_TO_ABB.get(b, b)
        ia, ib = DAY_SEQ.index(a), DAY_SEQ.index(b)
        if ia <= ib:
            out.update(DAY_SEQ[ia : ib + 1])
        else:
            out.update(DAY_SEQ[ia:] + DAY_SEQ[: ib + 1])

    # Single day names
    for token in re.findall(
        r"\b(MON|TUE|WED|THU|FRI|SAT|SUN|MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)\b",
        t,
    ):
        token = token.upper()
        out.add(DAY_NAME_TO_ABB.get(token, token))

    return out

def extract_duration(text: str) -> Set[str]:
    """
    Map common sign patterns to canonical tokens:
    - '2P' or 'P 2' -> '2 HOUR'
    - '30 MIN', '30 MINUTE(S)' -> '30 MIN'
    - '2 HOUR(S)', '2 HR(S)' -> '2 HOUR'
    """
    t = up(text).replace("–", "-").replace("—", "-")
    out = set()

    # '2P' or 'P 2'
    for m in re.finditer(r"\b(\d{1,3})\s*P\b", t, flags=re.I):
        out.add(f"{int(m.group(1))} HOUR")
    for m in re.finditer(r"\bP[\s\-]*(\d{1,3})\b", t, flags=re.I):
        out.add(f"{int(m.group(1))} HOUR")

    # Minutes
    for m in re.finditer(r"\b(\d{1,3})[\s\-]*(MIN(?:UTE)?S?)\b", t, flags=re.I):
        out.add(f"{int(m.group(1))} MIN")

    # Hours
    for m in re.finditer(r"\b(\d{1,3})[\s\-]*(H(?:OUR|R)?S?)\b", t, flags=re.I):
        out.add(f"{int(m.group(1))} HOUR")

    return out

def extract_times(text: str) -> Set[str]:
    """Collect single time tokens and endpoints of time ranges (start/end)."""
    t = up(text)
    out = set()

    # Single tokens
    for m in TIME_RE.finditer(t):
        out.add(normalize_time(m.group(1), m.group(2), m.group(3)))

    # Ranges: capture both ends
    for m in re.finditer(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(AM|PM)\s*-\s*(\d{1,2})(?::(\d{2}))?\s*(AM|PM)\b",
        t, flags=re.I
    ):
        out.add(normalize_time(m.group(1), m.group(2), m.group(3)))
        out.add(normalize_time(m.group(4), m.group(5), m.group(6)))

    return out

# -----------------------------------------------------
# Structured-first parsing from record["text"]
# -----------------------------------------------------
def parse_structured(rec: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Prefer values under rec["text"]:
      - days: str or list[str]
      - time: list of [start, end] strings (we take endpoints)
      - duration: free text (e.g., '2-hour limit')
    """
    text_obj = rec.get("text") if isinstance(rec, dict) else None
    if not isinstance(text_obj, dict):
        return {"days": set(), "times": set(), "duration": set()}

    days_set, times_set, dur_set = set(), set(), set()

    # days: str or list[str]
    days_field = text_obj.get("days")
    if isinstance(days_field, str):
        days_set |= extract_days(days_field)
    elif isinstance(days_field, list):
        for d in days_field:
            if isinstance(d, str):
                days_set |= extract_days(d)

    # time: list of ranges like [ ["9:00 AM","10:00 PM"], ... ]
    times_field = text_obj.get("time")
    if isinstance(times_field, list):
        for seg in times_field:
            if isinstance(seg, (list, tuple)):
                for tok in seg[:2]:
                    if isinstance(tok, str):
                        m = TIME_RE.search(up(tok))
                        if m:
                            times_set.add(normalize_time(m.group(1), m.group(2), m.group(3)))

    # duration: free text
    dur_field = text_obj.get("duration")
    if isinstance(dur_field, str):
        dur_set |= extract_duration(dur_field)

    return {"days": days_set, "times": times_set, "duration": dur_set}

# -----------------------------------------------------
# Natural-language fallback (optional, conservative)
# -----------------------------------------------------
def record_to_text_blob(rec: Dict[str, Any], use_nl: bool = False) -> str:
    """
    Concatenate structured fields under 'text' first.
    Only include broader natural language when use_nl=True.
    """
    chunks: List[str] = []

    def _add(v):
        if v is None:
            return
        if isinstance(v, (list, tuple)):
            for x in v:
                _add(x)
        elif isinstance(v, dict):
            # Prefer 'text' subtree
            if "text" in v and isinstance(v["text"], dict):
                _add(v["text"].get("days"))
                _add(v["text"].get("time"))
                _add(v["text"].get("duration"))
                _add(v["text"].get("rules"))
                _add(v["text"].get("direction"))
            elif use_nl and "natural_language" in v:
                _add(v["natural_language"])
        else:
            if use_nl:
                chunks.append(str(v))

    _add(rec)
    return " | ".join(chunks)

def extract_all(rec: Dict[str, Any], use_nl: bool = False) -> Dict[str, Set[str]]:
    """Use structured fields first; if all empty, optionally fall back to NL extraction."""
    s = parse_structured(rec)
    if any(s[k] for k in ("days","times","duration")):
        return s
    blob = record_to_text_blob(rec, use_nl=use_nl)
    return {
        "days": extract_days(blob),
        "duration": extract_duration(blob),
        "times": extract_times(blob),
    }

# -----------------------------------------------------
# JSON loader (flatten)
# -----------------------------------------------------
def load_flat(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat: List[Dict[str, Any]] = []
    if isinstance(data, list):
        nested = any(("parking_signs" in x) for x in data if isinstance(x, dict))
        if nested:
            for img in data:
                img_id = img.get("image") or img.get("image_id") or ""
                for s in img.get("parking_signs", []):
                    flat.append({"image_id": img_id, "sign_id": s.get("id", 0), **s})
        else:
            flat = data
    elif isinstance(data, dict):
        if "parking_signs" in data:
            img_id = data.get("image") or data.get("image_id") or ""
            for s in data.get("parking_signs", []):
                flat.append({"image_id": img_id, "sign_id": s.get("id", 0), **s})
        else:
            flat = [data]
    return flat

# -----------------------------------------------------
# Matching & scoring
# -----------------------------------------------------
def score_pair(gt_set: Set[str], pr_set: Set[str]) -> Tuple[int,int,int]:
    tp = len(gt_set & pr_set)
    fp = len(pr_set - gt_set)
    fn = len(gt_set - pr_set)
    return tp, fp, fn

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def jaccard(a: Set[str], b: Set[str]) -> float:
    inter = len(a & b)
    union = len(a | b)
    return (inter / union) if union > 0 else 0.0

def pair_score(gt_feats: Dict[str, Set[str]], pr_feats: Dict[str, Set[str]]) -> float:
    return (jaccard(gt_feats["days"], pr_feats["days"]) +
            jaccard(gt_feats["duration"], pr_feats["duration"]) +
            jaccard(gt_feats["times"], pr_feats["times"])) / 3.0

def match_greedy(gt_list: List[Dict[str, Any]],
                 pr_list: List[Dict[str, Any]],
                 use_nl: bool) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    """Greedy one-to-one matching by average Jaccard across three categories."""
    cand = []
    cached_gt = [extract_all(g, use_nl=use_nl) for g in gt_list]
    cached_pr = [extract_all(p, use_nl=use_nl) for p in pr_list]
    for i, gF in enumerate(cached_gt):
        for j, pF in enumerate(cached_pr):
            cand.append((pair_score(gF, pF), i, j))
    cand.sort(reverse=True)

    used_g, used_p, pairs = set(), set(), []
    for score, i, j in cand:
        if i in used_g or j in used_p:
            continue
        pairs.append((i, j))
        used_g.add(i); used_p.add(j)

    unmatched_g = [i for i in range(len(gt_list)) if i not in used_g]
    unmatched_p = [j for j in range(len(pr_list)) if j not in used_p]
    return pairs, unmatched_g, unmatched_p

# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Path to ground_truth.json")
    ap.add_argument("--pred", required=True, help="Path to prediction JSON")
    ap.add_argument("--per_image_report", action="store_true", help="Print per-image details")

    # Controls
    ap.add_argument("--ignore_nl", action="store_true",
                    help="Evaluate on structured text only; ignore natural_language fallback.")
    ap.add_argument("--gamma_fp_on_miss", type=float, default=0.30,
                    help="When FN>0 on a matched pair, add ceil(gamma*FN) FP as a mild precision tightening.")
    ap.add_argument("--fp_on_unmatched_pred", type=int, default=1,
                    help="FP added per unmatched predicted sign in an image (structural FP).")
    args = ap.parse_args()

    gt_rows = load_flat(args.gt)
    pr_rows = load_flat(args.pred)

    # Group by image_id
    img_ids = sorted(set([r.get("image_id", "") for r in gt_rows + pr_rows]))
    totals = {k: collections.Counter() for k in ["days", "duration", "times"]}

    print("=== Evaluation (categories: days / duration / times) ===")
    for img_id in img_ids:
        g_list = [r for r in gt_rows if (r.get("image_id", "") == img_id)]
        p_list = [r for r in pr_rows if (r.get("image_id", "") == img_id)]

        pairs, um_g, um_p = match_greedy(g_list, p_list, use_nl=(not args.ignore_nl))

        # Matched pairs: token-level scoring + mild precision tightening on misses
        for (i, j) in pairs:
            gtF = extract_all(g_list[i], use_nl=(not args.ignore_nl))
            prF = extract_all(p_list[j], use_nl=(not args.ignore_nl))

            if args.per_image_report:
                print(f"\n-- Image: {img_id} | Pair (GT#{i} vs PR#{j})")

            for cat in ["days", "duration", "times"]:
                tp, fp, fn = score_pair(gtF[cat], prF[cat])
                if fn > 0:
                    add_fp = int((args.gamma_fp_on_miss * fn + 0.9999))  # ceil
                    fp += add_fp

                totals[cat]["tp"] += tp
                totals[cat]["fp"] += fp
                totals[cat]["fn"] += fn

                if args.per_image_report:
                    P, R, F = prf(tp, fp, fn)
                    print(f"  [{cat}] GT={sorted(gtF[cat])}  PR={sorted(prF[cat])}  -> P={P:.3f} R={R:.3f} F1={F:.3f}")

        # Unmatched GT -> pure FN (missed signs)
        for i in um_g:
            gtF = extract_all(g_list[i], use_nl=(not args.ignore_nl))
            if args.per_image_report:
                print(f"\n-- Image: {img_id} | Unmatched GT#{i} (missed sign)")
            for cat in ["days", "duration", "times"]:
                totals[cat]["fn"] += len(gtF[cat])

        # Unmatched Pred -> structural FP (+ token-level FP if any)
        for j in um_p:
            prF = extract_all(p_list[j], use_nl=(not args.ignore_nl))
            if args.per_image_report:
                print(f"\n-- Image: {img_id} | Unmatched PR#{j} (extra sign)")
            for cat in ["days", "duration", "times"]:
                totals[cat]["fp"] += args.fp_on_unmatched_pred
                totals[cat]["fp"] += len(prF[cat])  # token-level FP for any extracted tokens

    # Summary
    print("\n=== Summary ===")
    micro_tp = micro_fp = micro_fn = 0
    for cat in ["days", "duration", "times"]:
        tp, fp, fn = totals[cat]["tp"], totals[cat]["fp"], totals[cat]["fn"]
        P, R, F = prf(tp, fp, fn)
        micro_tp += tp; micro_fp += fp; micro_fn += fn
        print(f"{cat:>8}:  Precision={P:.3f}  Recall={R:.3f}  F1={F:.3f}  (TP={tp} FP={fp} FN={fn})")

    P, R, F = prf(micro_tp, micro_fp, micro_fn)
    print(f"\nMicro-avg: Precision={P:.3f}  Recall={R:.3f}  F1={F:.3f}  (TP={micro_tp} FP={micro_fp} FN={micro_fn})")

if __name__ == "__main__":
    main()

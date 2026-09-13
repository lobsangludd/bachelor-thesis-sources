import json
import math
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
gold = {g["urn"]: g["label"] for g in json.load(open(ROOT_PATH / "gold" / "gold.json", encoding="utf-8"))}


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))

def clopper_pearson(k, n, alpha=0.05):
    def solve(target):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if target(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) < alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) >= alpha / 2)
    return lower, upper

def mcc(tp, fp, fn, tn):
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0

def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0

def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        if row["status"] == "ok":
            rows[row["article_urn"]] = row
    return {u: r for u, r in rows.items() if u in gold}

def score(rows):
    tp = sum(1 for u, r in rows.items() if gold[u] == "folk" and r["is_folktale"])
    fp = sum(1 for u, r in rows.items() if gold[u] != "folk" and r["is_folktale"])
    fn = sum(1 for u, r in rows.items() if gold[u] == "folk" and not r["is_folktale"])
    tn = len(rows) - tp - fp - fn
    fp_unsure = sum(1 for u, r in rows.items() if gold[u] == "unsure" and r["is_folktale"])
    tn_unsure = sum(1 for u, r in rows.items() if gold[u] == "unsure" and not r["is_folktale"])

    return {"N": len(rows), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall": (tp / (tp + fn), *clopper_pearson(tp, tp + fn)),
            "precision": (tp / (tp + fp) if tp + fp else 0.0, *clopper_pearson(tp, tp + fp)),
            "F1": 2 * tp / (2 * tp + fp + fn), "MCC": mcc(tp, fp, fn, tn),
            "MCC_no_unsure": mcc(tp, fp - fp_unsure, fn, tn - tn_unsure),
            "malformed": sum(1 for r in rows.values() if r.get("malformed_json"))}

runs = {Path(p).stem: load(p) for p in sys.argv[1:]}
table = sorted(((name, score(rows)) for name, rows in runs.items()), key=lambda x: -x[1]["MCC"])
width = max(len(name) for name in runs)

print(f"{'run':<{width}}     N  TP  FP  FN    TN  recall [95% CI]     precision [95% CI]     F1   MCC  MCC-unsure  malformed")

for name, stats in table:
    recall, recall_lo, recall_hi = stats["recall"]
    precision, precision_lo, precision_hi = stats["precision"]
    print(f"{name:<{width}} {stats['N']:>5} {stats['TP']:>3} {stats['FP']:>3} {stats['FN']:>3} {stats['TN']:>5}  "
          f"{recall:.3f} [{recall_lo:.2f}-{recall_hi:.2f}]   {precision:.3f} [{precision_lo:.2f}-{precision_hi:.2f}]  "
          f"{stats['F1']:.3f} {stats['MCC']:.3f}  {stats['MCC_no_unsure']:>10.3f}  {stats['malformed']:>9}")

if len(runs) == 2:
    (name_a, a), (name_b, b) = runs.items()
    shared = sorted(set(a) & set(b))
    right = lambda run, u: run[u]["is_folktale"] == (gold[u] == "folk")
    only_a = [u for u in shared if right(a, u) and not right(b, u)]
    only_b = [u for u in shared if right(b, u) and not right(a, u)]
    print(f"\nMcNemar over {len(shared)} shared articles: only {name_a} right {len(only_a)}, "
          f"only {name_b} right {len(only_b)}, p = {mcnemar(len(only_a), len(only_b)):.3f}")
    for u in shared:
        if a[u]["is_folktale"] != b[u]["is_folktale"]:
            print(f"  {u}  gold={gold[u]:<7} A={'folk' if a[u]['is_folktale'] else 'not':<5} "
                  f"B={'folk' if b[u]['is_folktale'] else 'not'}")

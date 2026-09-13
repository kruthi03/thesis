import csv, re
from collections import Counter
rows = list(csv.DictReader(open("out_narratives_ollama_v3/t085/labelling.csv", encoding="utf-8-sig")))
print("total rows:", len(rows))
labels = Counter(r["label"] for r in rows)
print("label value counts:", dict(labels))
rule_counts = Counter()
for r in rows:
    m = re.search(r"[Rr]ule (\d+) fired", r["notes"])
    if m:
        rule_counts[(m.group(1), r["label"])] += 1
print("(rule, label) -> count:")
for k in sorted(rule_counts): print(" ", k, rule_counts[k])

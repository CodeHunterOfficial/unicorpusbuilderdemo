import csv
from collections import defaultdict

# -------- правильная проверка рубрики --------
def is_valid_category(cat: str) -> bool:
    if not cat or not cat.strip():
        return False
    cat = cat.strip()
    # JSON-LD BreadcrumbList — это мусор, а не рубрика
    if cat.startswith("{"):
        return False
    # слишком длинная строка — не может быть названием рубрики
    if len(cat) > 100:
        return False
    return True

def is_valid(value: str) -> bool:
    return bool(value and value.strip())

# -------- агрегация --------
stats = defaultdict(lambda: {"n": 0, "author": 0, "date": 0, "cat": 0, "title": 0, "body": 0})

with open("trafilatura_comparison.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        site = row["site"].strip()
        if not site:
            continue
        # пропускаем ошибки fetch
        if row.get("error", "").strip():
            continue

        stats[site]["n"] += 1
        if is_valid(row.get("ucb_author", "")):
            stats[site]["author"] += 1
        if is_valid(row.get("ucb_date", "")):
            stats[site]["date"] += 1
        if is_valid_category(row.get("ucb_category", "")):
            stats[site]["cat"] += 1
        if is_valid(row.get("ucb_title", "")):
            stats[site]["title"] += 1
        if row.get("ucb_content_len", "0").strip() not in ("", "0"):
            stats[site]["body"] += 1

# -------- вывод --------
print(f"{'site':<18}{'N':>5}{'author':>10}{'date':>10}{'cat':>10}{'title':>10}{'body':>10}")
print("-" * 73)

total = {"n": 0, "author": 0, "date": 0, "cat": 0, "title": 0, "body": 0}
for site in sorted(stats):
    s = stats[site]
    n = s["n"]
    pct = lambda x: f"{100 * x / n:.1f}%" if n else "-"
    print(f"{site:<18}{n:>5}{pct(s['author']):>10}{pct(s['date']):>10}"
          f"{pct(s['cat']):>10}{pct(s['title']):>10}{pct(s['body']):>10}")
    for k in total:
        total[k] += s[k]

print("-" * 73)
n = total["n"]
pct = lambda x: f"{100 * x / n:.1f}%" if n else "-"
print(f"{'TOTAL':<18}{n:>5}{pct(total['author']):>10}{pct(total['date']):>10}"
      f"{pct(total['cat']):>10}{pct(total['title']):>10}{pct(total['body']):>10}")
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_all
from datetime import datetime
from collections import Counter

rows = get_mega645_all()
print(f"Total draws: {len(rows)}")
print(f"First: {rows[0]['draw_date']}: [{rows[0]['n1']},{rows[0]['n2']},{rows[0]['n3']},{rows[0]['n4']},{rows[0]['n5']},{rows[0]['n6']}]")
print(f"Last:  {rows[-1]['draw_date']}: [{rows[-1]['n1']},{rows[-1]['n2']},{rows[-1]['n3']},{rows[-1]['n4']},{rows[-1]['n5']},{rows[-1]['n6']}]")

dates = [datetime.strptime(r['draw_date'], '%Y-%m-%d') for r in rows]
gaps = []
for i in range(1, len(dates)):
    diff = (dates[i] - dates[i-1]).days
    if diff > 5:
        gaps.append((rows[i-1]['draw_date'], rows[i]['draw_date'], diff))

print(f"\nGaps > 5 days: {len(gaps)}")
for a, b, d in gaps[:10]:
    print(f"  {a} -> {b}: {d} days")

years = Counter(d.year for d in dates)
print(f"\nDraws per year:")
for y in sorted(years.keys()):
    print(f"  {y}: {years[y]}")

# Expected ~156/year (3 draws/week)
total_expected = sum(156 if y < 2026 else 156//4 for y in years.keys())
print(f"\nExpected total: ~{total_expected}")
print(f"Missing: ~{total_expected - len(rows)} draws")

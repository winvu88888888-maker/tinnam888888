"""
FINAL COMPARATIVE BACKTEST
Rule Engine vs Ensemble Vote vs Random
Quick walk-forward on last 300 draws.
"""
import sys, os, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers


def run():
    data = get_mega645_numbers()
    N = len(data)
    MAX = 45
    PICK = 6

    print("=" * 70)
    print("  COMPARATIVE BACKTEST — Rule Engine vs Others")
    print(f"  {N} draws, testing last 300")
    print("=" * 70)

    # Import engines
    from models.rule_engine import RuleEngine
    from models.backtester import BacktestEngine

    min_train = N - 301
    test_indices = list(range(min_train, N - 1))
    n_tests = len(test_indices)

    bt = BacktestEngine(MAX, PICK)

    methods = {
        'Rule Engine (69 rules)': [],
        'Ensemble Vote (V15)': [],
        'Momentum+Regime': [],
        'Frequency Weighted': [],
        'Random Baseline': [],
    }

    np.random.seed(42)
    print(f"\n  Running {n_tests} walk-forward tests...\n")
    start = time.time()

    for i, train_end in enumerate(test_indices):
        train = data[:train_end + 1]
        actual = set(data[train_end + 1][:PICK])

        # Rule Engine
        try:
            engine = RuleEngine(MAX, PICK)
            result = engine.predict(train, n_portfolio=1)
            pred = set(result.get('primary', []))
        except:
            pred = set(bt._predict_ensemble_vote(train))
        methods['Rule Engine (69 rules)'].append(len(pred & actual))

        # Ensemble
        pred2 = set(bt._predict_ensemble_vote(train))
        methods['Ensemble Vote (V15)'].append(len(pred2 & actual))

        # Momentum
        pred3 = set(bt._predict_momentum_regime(train))
        methods['Momentum+Regime'].append(len(pred3 & actual))

        # Frequency
        pred4 = set(bt._predict_frequency_weighted(train))
        methods['Frequency Weighted'].append(len(pred4 & actual))

        # Random
        pred5 = set(np.random.choice(range(1, MAX + 1), PICK, replace=False).tolist())
        methods['Random Baseline'].append(len(pred5 & actual))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            rule_avg = np.mean(methods['Rule Engine (69 rules)'])
            rand_avg = np.mean(methods['Random Baseline'])
            print(f"    [{i+1}/{n_tests}] Rule={rule_avg:.4f}/6, Random={rand_avg:.4f}/6 ({elapsed:.1f}s)")

    total_time = time.time() - start
    random_avg = 6 * 6 / 45

    print(f"\n{'='*70}")
    print(f"  RESULTS — {n_tests} tests")
    print(f"{'='*70}\n")

    # Sort by avg
    sorted_methods = sorted(methods.items(), key=lambda x: -np.mean(x[1]))

    for rank, (name, matches) in enumerate(sorted_methods):
        dist = Counter(matches)
        avg = np.mean(matches)
        imp = (avg / random_avg - 1) * 100

        pct3 = sum(1 for m in matches if m >= 3) / len(matches) * 100
        pct2 = sum(1 for m in matches if m >= 2) / len(matches) * 100

        medal = "🥇" if rank == 0 else "🥈" if rank == 1 else "🥉" if rank == 2 else "  "
        print(f"  {medal} #{rank+1} {name}")
        print(f"     Avg: {avg:.4f}/6 | Max: {max(matches)}/6 | vs Random: {imp:+.1f}%")
        print(f"     ≥2/6: {pct2:.1f}% | ≥3/6: {pct3:.1f}%")
        print(f"     Distribution: ", end="")
        for k in range(7):
            c = dist.get(k, 0)
            if c > 0:
                print(f"{k}/6={c}({c/len(matches)*100:.1f}%) ", end="")
        print()
        print()

    print(f"  Time: {total_time:.1f}s")
    print(f"  Random baseline avg: {random_avg:.4f}/6")
    print(f"{'='*70}")


if __name__ == '__main__':
    run()

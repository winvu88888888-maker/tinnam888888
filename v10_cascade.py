"""
V10 — SIÊU TRÍ TUỆ: CONSTRAINT CASCADE + SPACE COLLAPSE
=========================================================
TƯ DUY MỚI HOÀN TOÀN:
  V3-V9: Score 45 số → chọn top 6 → SAI (các số KHÔNG độc lập)
  V10: Predict CONSTRAINTS → Filter TOÀN BỘ 8M combos → còn bao nhiêu?

NẾU predict đúng:
  - Sum ±8 → giảm 60%
  - Odd count chính xác → giảm 80%  
  - Decade profile → giảm 90%
  - Min number ±3 → giảm 70%
  - Max number ±3 → giảm 70%
  
TỔNG: 8,145,060 → có thể còn VÀI NGÀN combo → mua HẾT!

PHASE 1: Xây XGBoost predict TỪNG constraint riêng biệt
PHASE 2: Đo accuracy & estimate "collapsed space" 
PHASE 3: Walk-forward test: enumerate & match
"""
import sys, os, math, json, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_mega645_all

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def build_draw_features(data, idx):
    """Features về DRAW (không phải từng số)."""
    if idx < 20:
        return None
    MAX, PICK = 45, 6
    curr = sorted(data[idx][:PICK])
    f = []

    # Draw properties (10)
    s = sum(curr)
    f.extend([s, max(curr)-min(curr), curr[0], curr[-1],
              sum(1 for x in curr if x%2==1), sum(1 for x in curr if x>22),
              np.mean(curr), np.std(curr), curr[2]-curr[1], curr[4]-curr[3]])

    # Previous draw properties (10)
    if idx >= 1:
        prev = sorted(data[idx-1][:PICK])
        sp = sum(prev)
        f.extend([sp, max(prev)-min(prev), prev[0], prev[-1],
                  sum(1 for x in prev if x%2==1), sum(1 for x in prev if x>22),
                  np.mean(prev), np.std(prev), prev[2]-prev[1], prev[4]-prev[3]])
    else:
        f.extend([0]*10)

    # Delta from prev (5)
    if idx >= 1:
        prev = sorted(data[idx-1][:PICK])
        f.extend([s - sum(prev), curr[0]-prev[0], curr[-1]-prev[-1],
                  (sum(1 for x in curr if x%2==1)) - (sum(1 for x in prev if x%2==1)),
                  len(set(curr) & set(prev))])
    else:
        f.extend([0]*5)

    # Previous 2 draws (5)
    if idx >= 2:
        prev2 = sorted(data[idx-2][:PICK])
        f.extend([sum(prev2), prev2[0], prev2[-1],
                  sum(1 for x in prev2 if x%2==1), len(set(curr) & set(prev2))])
    else:
        f.extend([0]*5)

    # Rolling stats (15)
    for w in [5, 10, 20]:
        window = data[max(0,idx-w+1):idx+1]
        ws = [sum(d[:PICK]) for d in window]
        wr = [max(d[:PICK])-min(d[:PICK]) for d in window]
        wmin = [min(d[:PICK]) for d in window]
        f.extend([np.mean(ws), np.std(ws), np.mean(wr), np.mean(wmin),
                  max(ws)-min(ws)])

    # Decade distribution (5)
    decs = Counter(min(x//10, 4) for x in curr)
    for d in range(5):
        f.append(decs.get(d, 0))

    # Digits pos1/pos6 (4)
    f.extend([curr[0]//10, curr[0]%10, curr[-1]//10, curr[-1]%10])

    # Each position value (6)
    f.extend(curr)

    return f


def run_v10():
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    dates = [r['draw_date'] for r in all_rows]
    N = len(data)
    MAX = 45
    PICK = 6
    C_TOTAL = math.comb(MAX, PICK)
    t0 = time.time()

    print("=" * 80)
    print("  🧠 V10 SIÊU TRÍ TUỆ — CONSTRAINT CASCADE + SPACE COLLAPSE")
    print(f"  {N} draws | Goal: collapse {C_TOTAL:,} → < 10,000 combos")
    print("=" * 80)

    # Define targets: properties of the NEXT draw
    def get_targets(draw):
        sd = sorted(draw[:PICK])
        return {
            'sum': sum(sd),
            'range': sd[-1] - sd[0],
            'min_num': sd[0],
            'max_num': sd[-1],
            'odd_count': sum(1 for x in sd if x % 2 == 1),
            'high_count': sum(1 for x in sd if x > 22),
            'pos2': sd[1],
            'pos5': sd[4],
            'decade_0': sum(1 for x in sd if x <= 9),
            'decade_1': sum(1 for x in sd if 10 <= x <= 19),
            'decade_2': sum(1 for x in sd if 20 <= x <= 29),
            'decade_3': sum(1 for x in sd if 30 <= x <= 39),
            'decade_4': sum(1 for x in sd if 40 <= x <= 45),
            'digit_min': sd[0] % 10,
            'digit_max': sd[-1] % 10,
        }

    # ================================================================
    # PHASE 1: Train property predictors
    # ================================================================
    print(f"\n{'━'*80}")
    print("  PHASE 1: Training Property Predictors")
    print(f"{'━'*80}")

    # Build features and targets
    X, Y = [], {}
    target_names = list(get_targets(data[0]).keys())
    for name in target_names:
        Y[name] = []

    for idx in range(20, N - 1):
        f = build_draw_features(data, idx)
        if f is None:
            continue
        X.append(f)
        tgt = get_targets(data[idx + 1])
        for name in target_names:
            Y[name].append(tgt[name])

    X = np.array(X)
    n_samples = len(X)
    train_end = int(n_samples * 0.7)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:train_end])
    X_test = scaler.transform(X[train_end:])
    n_test = len(X_test)

    print(f"  Samples: {n_samples}, Train: {train_end}, Test: {n_test}")
    print(f"  Features: {X.shape[1]}, Targets: {len(target_names)}")

    # Train regressors for continuous targets
    continuous = ['sum', 'range', 'min_num', 'max_num', 'pos2', 'pos5']
    # Train classifiers for discrete targets
    discrete = ['odd_count', 'high_count', 'decade_0', 'decade_1',
                'decade_2', 'decade_3', 'decade_4', 'digit_min', 'digit_max']

    models = {}
    accuracy = {}

    for name in target_names:
        y_train = np.array(Y[name][:train_end])
        y_test = np.array(Y[name][train_end:])

        if name in continuous:
            model = GradientBoostingRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Measure: what % of time actual is within ±K of prediction?
            for margin in [3, 5, 8, 10]:
                within = sum(1 for p, a in zip(preds, y_test) if abs(p - a) <= margin)
                pct = within / n_test * 100
                if margin == 5:
                    accuracy[name] = {'margin': margin, 'pct': round(pct, 1)}
            
            mae = np.mean(np.abs(preds - y_test))
            print(f"  {name:12s}: MAE={mae:.2f} | "
                  f"±3: {sum(1 for p,a in zip(preds,y_test) if abs(p-a)<=3)/n_test*100:.1f}% | "
                  f"±5: {sum(1 for p,a in zip(preds,y_test) if abs(p-a)<=5)/n_test*100:.1f}% | "
                  f"±8: {sum(1 for p,a in zip(preds,y_test) if abs(p-a)<=8)/n_test*100:.1f}%")
            models[name] = ('reg', model)
        else:
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = sum(1 for p, a in zip(preds, y_test) if p == a) / n_test * 100

            # Top-2 accuracy
            probs = model.predict_proba(X_test)
            classes = model.classes_
            top2_hits = 0
            for i in range(n_test):
                top2_idx = np.argsort(-probs[i])[:2]
                top2_vals = set(classes[j] for j in top2_idx)
                if y_test[i] in top2_vals:
                    top2_hits += 1
            top2_acc = top2_hits / n_test * 100

            accuracy[name] = {'exact': round(acc, 1), 'top2': round(top2_acc, 1)}
            print(f"  {name:12s}: exact={acc:.1f}% | top2={top2_acc:.1f}%")
            models[name] = ('clf', model)

    # ================================================================
    # PHASE 2: Estimate Space Collapse
    # ================================================================
    print(f"\n{'━'*80}")
    print("  PHASE 2: Space Collapse Estimation")
    print(f"{'━'*80}")

    # For each test draw: predict constraints → count valid combos
    space_sizes = []
    match_in_space = []

    for i in range(min(50, n_test)):  # Sample 50 test draws
        constraints = {}

        for name, (mtype, model) in models.items():
            if mtype == 'reg':
                pred = model.predict(X_test[i:i+1])[0]
                margin = 12 if name == 'sum' else 8
                constraints[name] = ('range', pred - margin, pred + margin)
            else:
                probs = model.predict_proba(X_test[i:i+1])[0]
                classes = model.classes_
                top3_idx = np.argsort(-probs)[:3]
                top2 = set(int(classes[j]) for j in top3_idx)
                constraints[name] = ('set', top2)

        # Count combos matching ALL constraints (sampling)
        aidx = min(train_end + 21 + i + 1, N - 1)
        actual = sorted(data[aidx][:PICK])
        actual_props = get_targets(actual)

        n_sample = 100000
        valid = 0
        actual_valid = 0
        np.random.seed(i)

        for _ in range(n_sample):
            combo = sorted(np.random.choice(range(1, MAX+1), PICK, replace=False))
            props = get_targets(combo)

            passes = True
            for name, constraint in constraints.items():
                if constraint[0] == 'range':
                    lo, hi = constraint[1], constraint[2]
                    if props[name] < lo or props[name] > hi:
                        passes = False
                        break
                else:  # set
                    if props[name] not in constraint[1]:
                        passes = False
                        break

            if passes:
                valid += 1

        # Check if actual draw passes
        actual_passes = True
        for name, constraint in constraints.items():
            if constraint[0] == 'range':
                lo, hi = constraint[1], constraint[2]
                if actual_props[name] < lo or actual_props[name] > hi:
                    actual_passes = False
                    break
            else:
                if actual_props[name] not in constraint[1]:
                    actual_passes = False
                    break

        estimated_space = int(C_TOTAL * valid / n_sample)
        space_sizes.append(estimated_space)
        match_in_space.append(1 if actual_passes else 0)

        if (i + 1) % 10 == 0:
            avg_space = np.mean(space_sizes)
            hit_rate = np.mean(match_in_space) * 100
            print(f"    [{i+1}/50] avg_space={avg_space:,.0f}, "
                  f"actual_in_space={hit_rate:.0f}%")

    avg_space = np.mean(space_sizes)
    containment = np.mean(match_in_space) * 100
    collapse_ratio = avg_space / C_TOTAL * 100

    print(f"\n  ═══════════════════════════════════════════")
    print(f"  Original space:    {C_TOTAL:>12,} combos")
    print(f"  After constraints: {avg_space:>12,.0f} combos (avg)")
    print(f"  Collapse ratio:    {collapse_ratio:.2f}%")
    print(f"  Actual in space:   {containment:.1f}% of time")
    print(f"  ═══════════════════════════════════════════")

    if avg_space < 50000:
        print(f"  🔴 SPACE < 50K! Feasible to enumerate & buy all!")
    elif avg_space < 500000:
        print(f"  🟡 Space 50K-500K. Need better constraints.")
    else:
        print(f"  ❌ Space still too large ({avg_space:,.0f})")

    # P(6/6) estimate
    if containment > 0:
        # If actual is in space X% of time, and space has S combos,
        # then P(6/6 | buy all in space) = containment%
        # But buying S tickets costs S × 10,000 VND
        tickets = int(avg_space)
        cost = tickets * 10000
        p_hit = containment / 100
        print(f"\n  Strategy: Buy ALL {tickets:,} tickets in collapsed space")
        print(f"  Cost per draw: {cost/1e6:.0f}M VND ({cost/1e9:.2f} tỷ)")
        print(f"  P(6/6 per draw): {p_hit*100:.1f}%")
        print(f"  Expected draws to hit: {1/max(p_hit,0.001):.0f}")
        print(f"  Expected cost to win: {cost/1e9 / max(p_hit,0.001):.1f} tỷ")

    # ================================================================
    # PHASE 3: Walk-forward with constraint filtering
    # ================================================================
    print(f"\n{'━'*80}")
    print("  PHASE 3: Walk-Forward — Smart Portfolio from Constraints")
    print(f"{'━'*80}")

    # For each test draw:
    # 1. Predict constraints
    # 2. Generate smart portfolio from predicted pool
    # 3. Score by how many constraints each combo satisfies

    port_sizes = [50, 500, 2000, 5000]
    results = {ps: [] for ps in port_sizes}
    random_results = []
    np.random.seed(42)

    for i in range(n_test):
        aidx = min(train_end + 21 + i + 1, N - 1)
        actual = set(data[aidx][:PICK])

        # Get per-number probabilities from constraint models
        num_scores = {num: 0.0 for num in range(1, MAX+1)}

        # Use regression predictions to score numbers
        for name, (mtype, model) in models.items():
            if mtype == 'reg':
                pred = model.predict(X_test[i:i+1])[0]
                for num in range(1, MAX+1):
                    sd_template = sorted([num] + [22]*5)  # rough
                    # Score numbers near predicted constraints
                    if name == 'min_num':
                        dist = abs(num - pred)
                        num_scores[num] += max(0, 5 - dist) * 0.5
                    elif name == 'max_num':
                        dist = abs(num - pred)
                        num_scores[num] += max(0, 5 - dist) * 0.5
                    elif name == 'pos2':
                        dist = abs(num - pred)
                        num_scores[num] += max(0, 4 - dist) * 0.3
                    elif name == 'pos5':
                        dist = abs(num - pred)
                        num_scores[num] += max(0, 4 - dist) * 0.3
            else:
                probs_clf = model.predict_proba(X_test[i:i+1])[0]
                classes = model.classes_
                if name == 'digit_min':
                    for j, c in enumerate(classes):
                        for num in range(1, MAX+1):
                            if num % 10 == c:
                                num_scores[num] += probs_clf[j] * 2
                elif name == 'digit_max':
                    for j, c in enumerate(classes):
                        for num in range(1, MAX+1):
                            if num % 10 == c:
                                num_scores[num] += probs_clf[j] * 2

        # Get sum/range predictions for filtering
        pred_sum = models['sum'][1].predict(X_test[i:i+1])[0]
        pred_range = models['range'][1].predict(X_test[i:i+1])[0]
        pred_min = models['min_num'][1].predict(X_test[i:i+1])[0]
        pred_max = models['max_num'][1].predict(X_test[i:i+1])[0]

        # Get odd count prediction
        odd_probs = models['odd_count'][1].predict_proba(X_test[i:i+1])[0]
        odd_classes = models['odd_count'][1].classes_
        top2_odd = set(int(odd_classes[j]) for j in np.argsort(-odd_probs)[:2])

        # Generate portfolio with CONSTRAINT filtering
        ranked = sorted(num_scores.items(), key=lambda x: -x[1])
        pool = [num for num, _ in ranked[:25]]

        portfolio = []
        seen = set()
        for combo in combinations(pool[:20], PICK):
            s = sum(combo)
            if abs(s - pred_sum) > 15:
                continue
            r = combo[-1] - combo[0]
            if abs(r - pred_range) > 8:
                continue
            if abs(combo[0] - pred_min) > 5:
                continue
            if abs(combo[-1] - pred_max) > 5:
                continue
            odds = sum(1 for x in combo if x % 2 == 1)
            if odds not in top2_odd:
                continue
            sc = sum(num_scores.get(n, 0) for n in combo)
            key = tuple(combo)
            if key not in seen:
                seen.add(key)
                portfolio.append((sorted(combo), sc))
            if len(portfolio) >= max(port_sizes) * 2:
                break

        portfolio.sort(key=lambda x: -x[1])

        for ps in port_sizes:
            port = [p[0] for p in portfolio[:ps]]
            if port:
                best = max(len(set(p) & actual) for p in port)
            else:
                best = 0
            results[ps].append(best)

        rand = set(np.random.choice(range(1, MAX+1), PICK, replace=False).tolist())
        random_results.append(len(rand & actual))

        if (i+1) % 100 == 0:
            print(f"    [{i+1}/{n_test}]")

    # ================================================================
    # RESULTS
    # ================================================================
    elapsed = time.time() - t0
    rand_avg = np.mean(random_results)

    print(f"\n{'='*80}")
    print(f"  📊 V10 RESULTS — {n_test} tests")
    print(f"{'='*80}")

    print(f"\n  {'Port':>6} | {'Avg':>6} | {'≥3':>6} | {'≥4':>6} | {'≥5':>6} | {'6/6':>6}")
    print(f"  {'─'*6} | {'─'*6} | {'─'*6} | {'─'*6} | {'─'*6} | {'─'*6}")
    print(f"  {'Rand':>6} | {rand_avg:.3f} | "
          f"{sum(1 for m in random_results if m>=3)/n_test*100:5.1f}% | "
          f"{sum(1 for m in random_results if m>=4)/n_test*100:5.1f}% | "
          f"{sum(1 for m in random_results if m>=5)/n_test*100:5.1f}% | "
          f"{sum(1 for m in random_results if m>=6)/n_test*100:5.2f}%")

    for ps in port_sizes:
        bm = results[ps]
        avg = np.mean(bm)
        print(f"  {ps:6d} | {avg:.3f} | "
              f"{sum(1 for m in bm if m>=3)/n_test*100:5.1f}% | "
              f"{sum(1 for m in bm if m>=4)/n_test*100:5.1f}% | "
              f"{sum(1 for m in bm if m>=5)/n_test*100:5.1f}% | "
              f"{sum(1 for m in bm if m>=6)/n_test*100:5.2f}%")

    # Save
    output = {
        'version': '10.0 — Constraint Cascade',
        'space_collapse': {
            'original': C_TOTAL,
            'collapsed_avg': int(avg_space),
            'collapse_pct': round(collapse_ratio, 2),
            'containment_pct': round(containment, 1),
        },
        'property_accuracy': accuracy,
        'results': {},
        'elapsed': round(elapsed, 1),
    }
    for ps in port_sizes:
        bm = results[ps]
        output['results'][f'port_{ps}'] = {
            'avg': round(np.mean(bm), 4),
            'pct_3': round(sum(1 for m in bm if m>=3)/n_test*100, 2),
            'pct_4': round(sum(1 for m in bm if m>=4)/n_test*100, 2),
            'pct_5': round(sum(1 for m in bm if m>=5)/n_test*100, 2),
            'pct_6': round(sum(1 for m in bm if m>=6)/n_test*100, 2),
        }
    path = os.path.join(os.path.dirname(__file__), 'models', 'v10_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved to {path}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_v10()

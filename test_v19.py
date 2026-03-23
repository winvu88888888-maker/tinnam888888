"""Backtest V19 — Full test on Mega 6/45 + Power 6/55 data."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.master_predictor import MasterPredictor
from collections import Counter
import numpy as np

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    draws = []
    for entry in data:
        nums = entry.get('numbers', entry.get('result', []))
        if isinstance(nums, list) and len(nums) >= 6:
            draws.append(nums[:6])
    return draws

def backtest_v19(draws, max_number, pick_count, name, test_count=300):
    """Walk-forward backtest: for each test point, predict using only past data."""
    n = len(draws)
    start = max(80, n - test_count)
    
    print(f"\n{'='*60}")
    print(f"BACKTEST V19 — {name} ({max_number}/{pick_count})")
    print(f"Total draws: {n}, Testing: {n - start - 1} draws")
    print(f"{'='*60}")
    
    matches_list = []
    portfolio_best_list = []
    
    predictor = MasterPredictor(max_number, pick_count)
    
    t0 = time.time()
    
    for i in range(start, n - 1):
        history = draws[:i+1]
        actual = set(draws[i+1])
        
        try:
            result = predictor.predict(history)
            pred = set(result['numbers'])
            match_count = len(pred & actual)
            matches_list.append(match_count)
            
            # Portfolio best
            portfolio = result.get('portfolio', [])
            if portfolio:
                best_p = max(len(set(p) & actual) for p in portfolio)
            else:
                best_p = match_count
            portfolio_best_list.append(best_p)
            
            elapsed = time.time() - t0
            avg_per = elapsed / len(matches_list)
            remaining = avg_per * (n - 1 - i - 1)
            
            if len(matches_list) % 10 == 0 or match_count >= 3:
                marker = " ⭐⭐⭐" if match_count >= 4 else (" ⭐" if match_count >= 3 else "")
                p_marker = " 🎯" if best_p >= 4 else ""
                print(f"  [{len(matches_list)}/{n-start-1}] Match: {match_count}/6{marker} | Portfolio best: {best_p}/6{p_marker} | ETA: {remaining:.0f}s")
        except Exception as e:
            print(f"  [{i-start+1}] ERROR: {e}")
            matches_list.append(0)
            portfolio_best_list.append(0)
    
    total_time = time.time() - t0
    total = len(matches_list)
    
    # Distribution
    dist = Counter(matches_list)
    p_dist = Counter(portfolio_best_list)
    
    # Random expected
    from math import comb
    random_prob = {}
    for k in range(pick_count + 1):
        random_prob[k] = comb(pick_count, k) * comb(max_number - pick_count, pick_count - k) / comb(max_number, pick_count)
    
    print(f"\n{'='*60}")
    print(f"RESULTS — {name} V19 ({total} tests, {total_time:.0f}s)")
    print(f"{'='*60}")
    
    print(f"\n📊 PRIMARY PREDICTION (best single set):")
    print(f"{'Match':>8} {'Count':>8} {'Rate':>10} {'Random':>10} {'vs Random':>10}")
    print(f"{'-'*50}")
    for k in range(6, -1, -1):
        cnt = dist.get(k, 0)
        rate = cnt / total * 100 if total > 0 else 0
        rand = random_prob.get(k, 0) * 100
        ratio = rate / rand if rand > 0 else 0
        marker = " ✅" if rate > rand * 1.3 else ""
        print(f"  {k}/6    {cnt:>6}    {rate:>7.2f}%    {rand:>7.2f}%    {ratio:>6.2f}x{marker}")
    
    avg = np.mean(matches_list)
    rexp = pick_count**2 / max_number
    print(f"\n  Average: {avg:.4f}/6 (random: {rexp:.4f})")
    print(f"  Improvement: {(avg/rexp - 1)*100:+.2f}%")
    print(f"  Max single: {max(matches_list)}/6")
    
    m3 = sum(1 for m in matches_list if m >= 3)
    m4 = sum(1 for m in matches_list if m >= 4)
    m5 = sum(1 for m in matches_list if m >= 5)
    m6 = sum(1 for m in matches_list if m >= 6)
    print(f"\n  >= 3 match: {m3} ({m3/total*100:.1f}%)")
    print(f"  >= 4 match: {m4} ({m4/total*100:.1f}%)")
    print(f"  >= 5 match: {m5} ({m5/total*100:.1f}%)")
    print(f"  == 6 match: {m6} ({m6/total*100:.1f}%)")
    
    if portfolio_best_list:
        print(f"\n📦 PORTFOLIO (best of 20 sets):")
        print(f"{'Match':>8} {'Count':>8} {'Rate':>10}")
        print(f"{'-'*30}")
        for k in range(6, -1, -1):
            cnt = p_dist.get(k, 0)
            rate = cnt / total * 100 if total > 0 else 0
            print(f"  {k}/6    {cnt:>6}    {rate:>7.2f}%")
        
        p_avg = np.mean(portfolio_best_list)
        print(f"\n  Portfolio avg best: {p_avg:.4f}/6")
        print(f"  Portfolio max: {max(portfolio_best_list)}/6")
        
        pm3 = sum(1 for m in portfolio_best_list if m >= 3)
        pm4 = sum(1 for m in portfolio_best_list if m >= 4)
        pm5 = sum(1 for m in portfolio_best_list if m >= 5)
        pm6 = sum(1 for m in portfolio_best_list if m >= 6)
        print(f"  >= 3: {pm3} ({pm3/total*100:.1f}%)")
        print(f"  >= 4: {pm4} ({pm4/total*100:.1f}%)")
        print(f"  >= 5: {pm5} ({pm5/total*100:.1f}%)")
        print(f"  == 6: {pm6} ({pm6/total*100:.1f}%)")
    
    return {
        'name': name, 'total': total,
        'avg': avg, 'max': max(matches_list),
        'distribution': dict(dist),
        'portfolio_avg': np.mean(portfolio_best_list) if portfolio_best_list else 0,
        'portfolio_max': max(portfolio_best_list) if portfolio_best_list else 0,
        'portfolio_dist': dict(p_dist),
    }

if __name__ == '__main__':
    # Mega 6/45
    mega_path = os.path.join(os.path.dirname(__file__), 'data', 'mega_results.json')
    if os.path.exists(mega_path):
        mega_draws = load_data(mega_path)
        print(f"Loaded Mega 6/45: {len(mega_draws)} draws")
        # Test with 100 draws for speed (full = 300 but very slow)
        test_count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
        backtest_v19(mega_draws, 45, 6, "Mega 6/45", test_count=test_count)
    else:
        print(f"No mega data at {mega_path}")

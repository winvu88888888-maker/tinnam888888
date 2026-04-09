"""
V15 DEEP ANALYSIS — TẠI SAO CHỈ 8/607 draws containment HIT 6/6?
==================================================================
Pool 40 chứa đúng 6 số trong 607/1286 draws = 47%.
Nhưng 50K tickets chỉ hit 8 draws = 8/607 = 1.3% of contained draws.

Theory: 50K/C(40,6) = 50000/3838380 = 1.3% → ĐÚNG!
→ Với pool 40, cần ~3.8M vé để cover 100% = KHÔNG THỰC TẾ.

MỤC TIÊU: Tìm cách THÔNG MINH hơn để chọn vé, không phải random sampling.
Nếu ta có thể RANK combos sao cho winning combo nằm trong top 1-5%,
thì 50K vé chỉ cần cover 1-5% = khả thi!

PHÂN TÍCH:
1. Kiểm tra 8 draws hit 6/6 — winning combo ranked ở vị trí nào?
2. Có pattern nào giúp rank winning combo cao hơn?
3. Nếu ta biết scoring function tốt → rank winning combo top 1% 
   → chỉ cần 38K vé bao phủ = ACHIEVABLE!
"""
import sys, os, math, json, time, random
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers

MAX = 45; PICK = 6


def compute_signals(data, at_index):
    relevant = data[:at_index]; n = len(relevant)
    if n < 50: return {num: PICK/MAX for num in range(1,MAX+1)}
    last = set(relevant[-1][:PICK]); base_p = PICK/MAX
    scores = {num: 0.0 for num in range(1,MAX+1)}
    for w, wt in [(3,4),(5,3),(10,2),(20,1.5),(50,1)]:
        if n < w: continue
        fc = Counter()
        for d in relevant[-w:]:
            for num in d[:PICK]: fc[num] += 1
        for num in range(1,MAX+1): scores[num] += fc.get(num,0)/w*wt
    follow = defaultdict(Counter); pc = Counter()
    for i in range(n-1):
        for p in relevant[i][:PICK]:
            pc[p] += 1
            for nx in relevant[i+1][:PICK]: follow[p][nx] += 1
    for num in range(1,MAX+1):
        tf = sum(follow[p].get(num,0) for p in last)
        tp = sum(pc[p] for p in last)
        if tp > 0: scores[num] += (tf/tp/base_p-1)*3
    ls_idx = {}; gs, gn = defaultdict(float), defaultdict(int)
    for i, d in enumerate(relevant):
        for num in d[:PICK]:
            if num in ls_idx: gs[num] += i-ls_idx[num]; gn[num] += 1
            ls_idx[num] = i
    for num in range(1,MAX+1):
        if gn[num] < 5: continue
        mg = gs[num]/gn[num]; cg = n-ls_idx.get(num,0)
        if cg > mg*1.3: scores[num] += min((cg/mg-1)*1.5, 3)
    knn = Counter()
    for i in range(n-2):
        sim = len(set(relevant[i][:PICK]) & last)
        if sim >= 2:
            for num in relevant[i+1][:PICK]: knn[num] += sim*sim
    mx = max(knn.values()) if knn else 1
    for num in range(1,MAX+1): scores[num] += knn.get(num,0)/mx*2.5
    if n >= 10:
        l1, l2 = set(relevant[-1][:PICK]), set(relevant[-2][:PICK])
        bc, ec = Counter(), Counter(); tb, te = 0, 0
        for i in range(2,n):
            p2,p1,cu = set(relevant[i-2][:PICK]),set(relevant[i-1][:PICK]),set(relevant[i][:PICK])
            for num in cu:
                if num in p2 and num in p1: bc[num]+=1
                elif num in p2 or num in p1: ec[num]+=1
            for num in range(1,MAX+1):
                if num in p2 and num in p1: tb+=1
                elif num in p2 or num in p1: te+=1
        for num in range(1,MAX+1):
            if num in l1 and num in l2: p=bc.get(num,0)/max(tb/MAX,1)
            elif num in l1 or num in l2: p=ec.get(num,0)/max(te/MAX,1)
            else: p=0
            scores[num]+=(p-base_p)*5
    pf = Counter()
    for d in relevant[-200:]:
        for p in combinations(sorted(d[:PICK]),2): pf[p]+=1
    for num in range(1,MAX+1):
        scores[num]+=sum(pf.get(tuple(sorted([p,num])),0) for p in last)*0.05
    tf = Counter()
    for d in relevant[-150:]:
        for t in combinations(sorted(d[:PICK]),3): tf[t]+=1
    for t, c in tf.most_common(500):
        if c < 2: break
        ts = set(t)
        if len(ts&last)==2: scores[(ts-last).pop()]+=c*0.3
    if n >= 100:
        rw = min(100,n//3); fr,fo = Counter(),Counter()
        for d in relevant[-rw:]:
            for num in d[:PICK]: fr[num]+=1
        for d in relevant[:-rw]:
            for num in d[:PICK]: fo[num]+=1
        for num in range(1,MAX+1):
            scores[num]+=(fr.get(num,0)/rw-fo.get(num,0)/max(n-rw,1))*10
    ww = min(100,n); iai,ic = Counter(),Counter()
    for i in range(max(0,n-ww),n-1):
        curr,nxt = set(relevant[i][:PICK]),set(relevant[i+1][:PICK])
        for num in curr:
            ic[num]+=1
            if num in nxt: iai[num]+=1
    for num in range(1,MAX+1):
        if num in last and ic[num]>5: scores[num]+=(iai[num]/ic[num]-base_p)*3
    return scores


def build_pool(data, at_index, scores, max_pool):
    relevant = data[:at_index]; n = at_index
    last_set = set(relevant[-1][:PICK])
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    pools = set()
    pools.update(num for num, _ in ranked[:15])
    for w in [30,50,100]:
        fc = Counter(num for d in relevant[-min(w,n):] for num in d[:PICK])
        pools.update(num for num, _ in fc.most_common(15))
    ls = {}
    for i, d in enumerate(relevant):
        for num in d[:PICK]: ls[num] = i
    pools.update(sorted(range(1,MAX+1), key=lambda x: -(n-ls.get(x,0)))[:15])
    fc = Counter()
    for i in range(n-1):
        for p in relevant[i][:PICK]:
            if p in last_set:
                for nx in relevant[i+1][:PICK]: fc[nx]+=1
    pools.update(num for num, _ in fc.most_common(15))
    kc = Counter()
    for i in range(n-2):
        sim = len(set(relevant[i][:PICK])&last_set)
        if sim>=3:
            for num in relevant[i+1][:PICK]: kc[num]+=sim*sim
    pools.update(num for num, _ in kc.most_common(15))
    cc = Counter()
    for d in relevant[-200:]:
        ds = set(d[:PICK]); ov = ds&last_set
        if len(ov)>=2:
            for num in ds-last_set: cc[num]+=len(ov)
    pools.update(num for num, _ in cc.most_common(15))
    for num in last_set:
        for delta in [-2,-1,1,2]:
            nb = num+delta
            if 1<=nb<=MAX: pools.add(nb)
    return sorted(pools, key=lambda x: -scores.get(x,0))[:max_pool]


def score_combo(combo, scores, pair_hist):
    """Score a specific combo."""
    sc = sum(scores.get(n,0) for n in combo)
    # Pair affinity
    for a,b in combinations(combo,2):
        sc += pair_hist.get(tuple(sorted([a,b])),0) * 0.01
    # Property fit
    s = sum(combo); r = combo[-1]-combo[0]; odd = sum(1 for x in combo if x%2==1)
    sc -= (abs(s-138)/50 + abs(r-30)/15 + abs(odd-3)/3) * 0.1
    return sc


def run_deep_analysis():
    data = get_mega645_numbers()
    N = len(data); t0 = time.time()
    
    print("="*80)
    print("  🔬 V15 DEEP ANALYSIS — WHY NOT 6/6 MORE OFTEN?")
    print(f"  {N} draws | Analyze winning combo ranking")
    print("="*80)
    
    WARMUP = 200; POOL_MAX = 40
    n_test = N - WARMUP
    
    # Track: for draws where pool contains all 6, WHERE does winning combo rank?
    contained_draws = []
    winning_ranks = []
    winning_percentiles = []
    not_contained = 0
    
    # Also: per-number rank analysis
    num_rank_in_pool = []  # rank of each winning number in pool
    
    print(f"\n  Analyzing {n_test} draws...")
    
    for ti in range(n_test):
        te = WARMUP + ti
        actual = set(data[te][:PICK])
        actual_sorted = sorted(actual)
        
        sc = compute_signals(data, te)
        pool = build_pool(data, te, sc, POOL_MAX)
        pool_set = set(pool)
        
        # Check containment
        if len(actual & pool_set) < 6:
            not_contained += 1
            continue
        
        # All 6 numbers IN pool!
        contained_draws.append(te)
        
        # Rank each winning number in pool
        for num in actual:
            rank = pool.index(num) + 1  # 1-indexed
            num_rank_in_pool.append(rank)
        
        # Build pair history
        pair_hist = Counter()
        for d in data[max(0,te-200):te]:
            for p in combinations(sorted(d[:PICK]),2): pair_hist[p]+=1
        
        # Score ALL combos from pool (sample if too large)
        pool_list = list(pool)
        total_combos = math.comb(len(pool_list), PICK)
        
        if total_combos <= 50000:
            # Enumerate all and find winning combo rank
            all_scores = []
            winning_score = None
            winning_combo = tuple(sorted(actual))
            
            for combo in combinations(pool_list, PICK):
                cs = score_combo(list(combo), sc, pair_hist)
                if tuple(sorted(combo)) == winning_combo:
                    winning_score = cs
                all_scores.append(cs)
            
            if winning_score is not None:
                all_scores.sort(reverse=True)
                rank = all_scores.index(winning_score) + 1
                pctl = rank / len(all_scores) * 100
                winning_ranks.append(rank)
                winning_percentiles.append(pctl)
        else:
            # Sample approach: score winning combo + random sample
            winning_combo = list(sorted(actual))
            ws = score_combo(winning_combo, sc, pair_hist)
            
            n_sample = 100000
            higher = 0
            for _ in range(n_sample):
                idx = random.sample(range(len(pool_list)), PICK)
                sample_combo = sorted([pool_list[i] for i in idx])
                ss = score_combo(sample_combo, sc, pair_hist)
                if ss > ws:
                    higher += 1
            
            pctl = higher / n_sample * 100
            est_rank = int(pctl / 100 * total_combos)
            winning_ranks.append(est_rank)
            winning_percentiles.append(pctl)
        
        if len(contained_draws) % 50 == 0:
            avg_pctl = np.mean(winning_percentiles)
            med_pctl = np.median(winning_percentiles)
            print(f"  [{len(contained_draws)} contained] avg_pctl={avg_pctl:.1f}% median={med_pctl:.1f}%")
            sys.stdout.flush()
    
    elapsed = time.time() - t0
    
    print(f"\n{'═'*80}")
    print(f"  🔬 DEEP ANALYSIS RESULTS")
    print(f"{'═'*80}")
    
    print(f"\n  Pool containment: {len(contained_draws)}/{n_test} ({len(contained_draws)/n_test*100:.1f}%)")
    print(f"  Not contained: {not_contained}")
    
    if winning_percentiles:
        print(f"\n  📊 WINNING COMBO PERCENTILE RANKING:")
        print(f"    Mean: {np.mean(winning_percentiles):.1f}%")
        print(f"    Median: {np.median(winning_percentiles):.1f}%")
        print(f"    Std: {np.std(winning_percentiles):.1f}%")
        print(f"    Min: {np.min(winning_percentiles):.1f}%")
        print(f"    Max: {np.max(winning_percentiles):.1f}%")
        
        # Histogram
        bins = [0,1,2,5,10,20,30,50,75,100]
        print(f"\n    Percentile distribution:")
        for i in range(len(bins)-1):
            count = sum(1 for p in winning_percentiles if bins[i]<=p<bins[i+1])
            pct = count/len(winning_percentiles)*100
            bar = '█' * int(pct)
            print(f"    {bins[i]:3d}-{bins[i+1]:3d}%: {count:4d} ({pct:5.1f}%) {bar}")
        
        # Key insight
        top1 = sum(1 for p in winning_percentiles if p <= 1)
        top5 = sum(1 for p in winning_percentiles if p <= 5)
        top10 = sum(1 for p in winning_percentiles if p <= 10)
        top20 = sum(1 for p in winning_percentiles if p <= 20)
        
        print(f"\n    ⚡ KEY NUMBERS:")
        print(f"    Winning combo in top 1%: {top1}/{len(winning_percentiles)} ({top1/len(winning_percentiles)*100:.1f}%)")
        print(f"    Winning combo in top 5%: {top5}/{len(winning_percentiles)} ({top5/len(winning_percentiles)*100:.1f}%)")
        print(f"    Winning combo in top 10%: {top10}/{len(winning_percentiles)} ({top10/len(winning_percentiles)*100:.1f}%)")
        print(f"    Winning combo in top 20%: {top20}/{len(winning_percentiles)} ({top20/len(winning_percentiles)*100:.1f}%)")
        
        # Implication
        total_combos_40 = math.comb(40, 6)
        print(f"\n    📐 IMPLICATIONS (Pool 40, C(40,6)={total_combos_40:,}):")
        for target_pct in [1, 5, 10, 20]:
            tickets_needed = int(total_combos_40 * target_pct / 100)
            hit_rate = sum(1 for p in winning_percentiles if p <= target_pct) / len(winning_percentiles)
            print(f"    Top {target_pct:2d}% = {tickets_needed:,} tickets → "
                  f"hit {hit_rate*100:.1f}% of contained draws → "
                  f"6/6 rate = {hit_rate * len(contained_draws)/n_test*100:.2f}%")
    
    if num_rank_in_pool:
        print(f"\n  📊 PER-NUMBER RANK IN POOL:")
        avg_nr = np.mean(num_rank_in_pool)
        print(f"    Avg rank: {avg_nr:.1f} / {POOL_MAX}")
        print(f"    Numbers in top 10: {sum(1 for r in num_rank_in_pool if r<=10)/len(num_rank_in_pool)*100:.1f}%")
        print(f"    Numbers in top 20: {sum(1 for r in num_rank_in_pool if r<=20)/len(num_rank_in_pool)*100:.1f}%")
        print(f"    Numbers in top 30: {sum(1 for r in num_rank_in_pool if r<=30)/len(num_rank_in_pool)*100:.1f}%")
    
    print(f"\n  Time: {elapsed:.0f}s")
    
    # Save
    output = {
        'analysis': 'winning_combo_ranking',
        'n_contained': len(contained_draws),
        'avg_percentile': round(np.mean(winning_percentiles),1) if winning_percentiles else None,
        'median_percentile': round(np.median(winning_percentiles),1) if winning_percentiles else None,
        'top1_pct': round(top1/len(winning_percentiles)*100,1) if winning_percentiles else 0,
        'top5_pct': round(top5/len(winning_percentiles)*100,1) if winning_percentiles else 0,
        'top10_pct': round(top10/len(winning_percentiles)*100,1) if winning_percentiles else 0,
        'elapsed': round(elapsed,1),
    }
    path = os.path.join(os.path.dirname(__file__), 'models', 'v15_deep_analysis.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


if __name__ == '__main__':
    run_deep_analysis()

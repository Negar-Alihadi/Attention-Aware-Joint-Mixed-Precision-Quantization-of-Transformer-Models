"""
Testing the ILP, Pareto-frontier, and validation logic in mckp_ilp_pareto.py.
Using small, hand-constructed sensitivity tables with KNOWN correct answers
(computable by hand / brute force), so failures point at a real bug rather
than needing GPT-2 or calibration data to debug.
"""
import itertools
from mckp_ilp_pareto import solve_mckp_ilp, solve_pareto_frontier, validate_allocators

def brute_force_optimal(scores, budget):
    names = list(scores.keys())
    bit_widths_per_block = [sorted(scores[n].keys()) for n in names]
    best_assignment, best_sensitivity = None, float("inf")
    for combo in itertools.product(*bit_widths_per_block):
        total_cost = sum(combo)
        if total_cost > budget:
            continue
        total_sens = sum(scores[names[i]][combo[i]] for i in range(len(names)))
        if total_sens < best_sensitivity:
            best_sensitivity = total_sens
            best_assignment = dict(zip(names, combo))
    return best_assignment, best_sensitivity

def test_ilp_matches_brute_force():
    print("\n=== Test 1: ILP vs. brute force (small, exact problem) ===")
    scores = {
        "block_0": {2: 10.0, 3: 5.0, 4: 2.0},
        "block_1": {2: 8.0,  3: 4.0, 4: 1.0},
        "block_2": {2: 12.0, 3: 6.0, 4: 3.0},
    }
    budget = 9.0

    bf_assignment, bf_sensitivity = brute_force_optimal(scores, budget)
    ilp_assignment, ilp_cost, ilp_sensitivity = solve_mckp_ilp(scores, budget)

    print(f"  Brute force: sensitivity={bf_sensitivity:.4f}  {bf_assignment}")
    print(f"  ILP: sensitivity={ilp_sensitivity:.4f}  {ilp_assignment}")

    assert abs(ilp_sensitivity - bf_sensitivity) < 1e-6
    assert ilp_cost <= budget
    print("  PASSED: ILP found the true optimum.")

def test_ilp_respects_budget():
    print("\n=== Test 2: ILP respects budget on a larger random problem ===")
    import random
    rng = random.Random(0)
    scores = {}
    for i in range(8):
        fragility = rng.uniform(0.5, 3.0)
        scores[f"block_{i}"] = {b: fragility / (b ** 1.5) for b in [2, 3, 4, 8, 16]}

    for budget in [16.0, 24.0, 32.0, 48.0]:
        assignment, cost, sensitivity = solve_mckp_ilp(scores, budget)
        print(f"  budget={budget:.1f}: cost_used={cost:.1f}, sensitivity={sensitivity:.4f}")
        assert cost <= budget + 1e-6
    print("  PASSED: ILP respects the budget at every tested level.")

def test_pareto_never_beats_ilp():
    print("\n=== Test 3: Pareto sensitivity >= ILP sensitivity (always) ===")
    import random
    rng = random.Random(1)
    scores = {}
    for i in range(10):
        fragility = rng.uniform(0.3, 4.0)
        scores[f"block_{i}"] = {b: fragility / (b ** 1.5) for b in [2, 3, 4, 8, 16]}

    for budget in [20.0, 30.0, 40.0]:
        _, _, ilp_sens = solve_mckp_ilp(scores, budget)
        _, _, pareto_sens = solve_pareto_frontier(scores, budget)
        print(f"  budget={budget:.1f}: ILP={ilp_sens:.4f}  Pareto={pareto_sens:.4f}  "
              f"gap={(pareto_sens - ilp_sens) / ilp_sens * 100:+.2f}%")
        assert pareto_sens >= ilp_sens - 1e-9
    print("  PASSED: Pareto never outperforms the exact optimum.")

def test_pareto_frontier_dominance_filtering():
    print("\n=== Test 4: dominated candidates are correctly filtered ===")
    from mckp_ilp_pareto import _pareto_frontier_for_block

    candidates = [(2, 10.0), (3, 4.0), (4, 5.0), (8, 1.0), (16, 0.5)]
    frontier = _pareto_frontier_for_block(candidates)
    print(f"  Input:    {candidates}")
    print(f"  Frontier: {frontier}")

    frontier_costs = [c for c, s in frontier]
    assert 4 not in frontier_costs
    assert 2 in frontier_costs and 3 in frontier_costs and 8 in frontier_costs and 16 in frontier_costs
    print("  PASSED: dominated candidate correctly excluded, useful ones kept.")

def test_validate_allocators_runs_and_agrees():
    print("\n=== Test 5: validate_allocators end-to-end ===")
    import random
    rng = random.Random(42)
    scores = {}
    for i in range(12):
        fragility = rng.uniform(0.5, 3.0)
        scores[f"block_{i}_QKV"] = {b: fragility / (b ** 1.5) for b in [2, 3, 4, 8, 16]}

    budget = 4.3 * 12
    results = validate_allocators(scores, budget, n_blocks_slice=4)

    ilp_sens = results["ilp"][2]
    pareto_sens = results["pareto"][2]
    greedy_sens = results["greedy"][2]

    pareto_gap_pct = (pareto_sens - ilp_sens) / ilp_sens * 100
    greedy_gap_pct = (greedy_sens - ilp_sens) / ilp_sens * 100

    assert pareto_gap_pct < 20
    assert greedy_gap_pct < 20
    print("  PASSED: both cheap allocators stay within a reasonable gap of exact ILP.")

if __name__ == "__main__":
    print("TESTING mckp_ilp_pareto.py")
    test_ilp_matches_brute_force()
    test_ilp_respects_budget()
    test_pareto_never_beats_ilp()
    test_pareto_frontier_dominance_filtering()
    test_validate_allocators_runs_and_agrees()
    print("\nALL TESTS PASSED!")

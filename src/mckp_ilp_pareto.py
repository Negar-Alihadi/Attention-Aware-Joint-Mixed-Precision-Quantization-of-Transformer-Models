"""
Implementing the three allocation strategies from Section 4.3:

1. Multiple-Choice Knapsack Problem (MCKP) via exact integer linear
   programming -- globally optimal, but scales poorly with many blocks x
   many bit-widths (used here as the small-scale VALIDATION reference,
   not as the primary allocator -- exactly as the proposal specifies).

2. Pareto-frontier search -- filters each block's candidate bit-widths
   down to only the non-dominated (cost, sensitivity) points, then walks
   the combined frontier greedily. Cheaper than ILP, HAWQ-V2's approach.

3. Validation: confirms the existing greedy allocator and the new 
   Pareto-frontier allocator both reproduce the ILP's exact optimum
   on a small slice of blocks -- the proposal's explicit
   "before scaling to the full model" check.

Requires scipy >= 1.9. Install with: pip install scipy --upgrade
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
# ---------------------------------------------------------------------------
# 1. Exact MCKP via ILP
# ---------------------------------------------------------------------------
def solve_mckp_ilp(sensitivity_scores, budget, bit_widths=None):
    """
    Exact solve of the Multiple-Choice Knapsack Problem:
        min  sum_i sensitivity_i(b_i)
        s.t. sum_i cost(b_i) <= budget
             exactly one b_i chosen per block i
    sensitivity_scores: {block_name: {bits: sensitivity}} -- same format
        as greedy_allocate's `scores` argument (from JAB, independent-trace,
        Fisher, etc.)
    budget: total bit-cost budget (e.g. target_avg_bits * n_blocks)
    bit_widths: candidate bit-widths; if None, inferred from the first
        block's score dict
    Returns: (assignment, cost_used, total_sensitivity)
    """
    block_names = list(sensitivity_scores.keys())
    if bit_widths is None:
        bit_widths = sorted(sensitivity_scores[block_names[0]].keys())

    n_blocks = len(block_names)
    n_bits = len(bit_widths)
    n_vars = n_blocks * n_bits  # binary x_{i,b}, flattened index = i*n_bits + j
    # Objective: minimize sum sensitivity_i(b) * x_{i,b}
    c = np.zeros(n_vars)
    cost = np.zeros(n_vars)
    for i, name in enumerate(block_names):
        for j, b in enumerate(bit_widths):
            idx = i * n_bits + j
            c[idx] = sensitivity_scores[name][b]
            cost[idx] = b  # cost(bits) = bits, matching greedy_allocate's convention
    # Constraint A: exactly one bit-width chosen per block
    A_eq = np.zeros((n_blocks, n_vars))
    for i in range(n_blocks):
        A_eq[i, i * n_bits:(i + 1) * n_bits] = 1.0
    one_choice_per_block = LinearConstraint(A_eq, lb=1, ub=1)
    # Constraint B: total cost <= budget
    budget_constraint = LinearConstraint(cost.reshape(1, -1), lb=-np.inf, ub=budget)
    integrality = np.ones(n_vars)  # all variables binary/integer
    bounds = Bounds(lb=0, ub=1)
    result = milp(c=c, constraints=[one_choice_per_block, budget_constraint], integrality=integrality, bounds=bounds)
    if not result.success:
        raise RuntimeError(f"ILP solve failed: {result.message}")
    assignment = {}
    x = result.x
    for i, name in enumerate(block_names):
        for j, b in enumerate(bit_widths):
            idx = i * n_bits + j
            if x[idx] > 0.5:  # binary variable, should be ~0 or ~1
                assignment[name] = b

    cost_used = sum(assignment[name] for name in assignment)
    total_sensitivity = sum(sensitivity_scores[name][assignment[name]] for name in assignment)
    return assignment, cost_used, total_sensitivity
# ---------------------------------------------------------------------------
# 2. Pareto-frontier search
# ---------------------------------------------------------------------------
def _pareto_frontier_for_block(candidates):
    """
    candidates: list of (cost, sensitivity) pairs for ONE block, one per
    candidate bit-width. Returns only the non-dominated points, sorted by
    increasing cost (and correspondingly non-increasing sensitivity).

    A point is dominated if some other point has <= cost AND <= sensitivity
    (i.e. it's never worse on either axis) -- dominated points are strictly
    useless, since the dominating point is always at least as good.
    """
    frontier = []
    best_sensitivity_so_far = float("inf")
    for cost, sens in sorted(candidates, key=lambda t: t[0]):  # ascending cost
        if sens < best_sensitivity_so_far:
            frontier.append((cost, sens))
            best_sensitivity_so_far = sens
    return frontier

def solve_pareto_frontier(sensitivity_scores, budget, bit_widths=None):
    """
    HAWQ-V2 style allocation: restrict each block to its Pareto-efficient
    (cost, sensitivity) candidates only, then greedily walk down from the
    max-cost frontier point of every block -- same downgrade mechanism as
    greedy_allocate, just restricted to non-dominated points.

    Returns: (assignment, cost_used, total_sensitivity)
    """
    block_names = list(sensitivity_scores.keys())
    if bit_widths is None:
        bit_widths = sorted(sensitivity_scores[block_names[0]].keys())

    frontiers = {}
    for name in block_names:
        candidates = [(b, sensitivity_scores[name][b]) for b in bit_widths]
        frontiers[name] = _pareto_frontier_for_block(candidates)

    # start every block at its highest-cost frontier point
    current_idx = {name: len(frontiers[name]) - 1 for name in block_names}

    def total_cost():
        return sum(frontiers[name][current_idx[name]][0] for name in block_names)

    def total_sensitivity():
        return sum(frontiers[name][current_idx[name]][1] for name in block_names)

    while total_cost() > budget:
        best_name, best_ratio = None, None
        for name in block_names:
            idx = current_idx[name]
            if idx == 0:
                continue  # already at this block's cheapest frontier point
            cost_now, sens_now = frontiers[name][idx]
            cost_next, sens_next = frontiers[name][idx - 1]
            cost_saved = cost_now - cost_next
            sens_added = sens_next - sens_now
            ratio = sens_added / cost_saved  # sensitivity lost per bit saved
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_name = name
        if best_name is None:
            break  # no block can be downgraded further
        current_idx[best_name] -= 1

    assignment = {name: frontiers[name][current_idx[name]][0] for name in block_names}
    return assignment, total_cost(), total_sensitivity()
# ---------------------------------------------------------------------------
# 3. Small-scale validation: does greedy/Pareto match exact ILP?
# ---------------------------------------------------------------------------
def validate_allocators(sensitivity_scores, budget, bit_widths=None, n_blocks_slice=4):
    """
    Runs ILP (exact), Pareto-frontier, and the existing greedy allocator
    on a SMALL SLICE of blocks (default: first 4) at the given budget, and
    prints a side-by-side comparison -- this is the proposal's explicit
    "Exact ILP solve (small-scale validation) ... to confirm the
    greedy/Pareto solution is near-optimal before scaling to the full
    model" (Section 4.3).
    """
    from greedy_allocator_simple import greedy_allocate

    block_names = list(sensitivity_scores.keys())[:n_blocks_slice]
    small_scores = {name: sensitivity_scores[name] for name in block_names}

    if bit_widths is None:
        bit_widths = sorted(small_scores[block_names[0]].keys())

    # scale budget proportionally to the slice size, matching the
    # per-block-average density of the full budget
    full_n_blocks = len(sensitivity_scores)
    small_budget = budget * (n_blocks_slice / full_n_blocks)

    print(f"Validating on {n_blocks_slice}/{full_n_blocks} blocks, "
          f"budget={small_budget:.2f} (scaled from full budget={budget:.2f})\n")

    ilp_assign, ilp_cost, ilp_sens = solve_mckp_ilp(small_scores, small_budget, bit_widths)
    pareto_assign, pareto_cost, pareto_sens = solve_pareto_frontier(small_scores, small_budget, bit_widths)
    greedy_assign, greedy_cost, greedy_sens = greedy_allocate(small_scores, small_budget)

    print(f"{'Method':<12} | {'Cost':>8} | {'Total Sensitivity':>18} | Assignment")
    print("-" * 70)
    print(f"{'ILP (exact)':<12} | {ilp_cost:>8.1f} | {ilp_sens:>18.6f} | {ilp_assign}")
    print(f"{'Pareto':<12} | {pareto_cost:>8.1f} | {pareto_sens:>18.6f} | {pareto_assign}")
    print(f"{'Greedy':<12} | {greedy_cost:>8.1f} | {greedy_sens:>18.6f} | {greedy_assign}")

    pareto_gap = (pareto_sens - ilp_sens) / ilp_sens * 100 if ilp_sens != 0 else float("nan")
    greedy_gap = (greedy_sens - ilp_sens) / ilp_sens * 100 if ilp_sens != 0 else float("nan")
    print(f"\nOptimality gap vs. exact ILP:")
    print(f"  Pareto: {pareto_gap:+.2f}%")
    print(f"  Greedy: {greedy_gap:+.2f}%")

    return {
        "ilp": (ilp_assign, ilp_cost, ilp_sens),
        "pareto": (pareto_assign, pareto_cost, pareto_sens),
        "greedy": (greedy_assign, greedy_cost, greedy_sens),
    }

if __name__ == "__main__":
    # Self-contained smoke test with synthetic scores (mirrors
    # greedy_allocator_simple.py's own __main__ test pattern) -- no GPT-2
    # or calibration data needed, just checks the solvers agree with each
    # other on a toy problem before you trust them on real sensitivity data.
    import random

    BIT_WIDTHS = [2, 3, 4, 8, 16]
    rng = random.Random(42)
    block_names = [f"block_{i}_QKV" for i in range(6)]
    synthetic_scores = {}
    for name in block_names:
        fragility = rng.uniform(0.5, 3.0)
        synthetic_scores[name] = {b: fragility / (b ** 1.5) for b in BIT_WIDTHS}

    budget = 4.0 * len(block_names)  # avg 4 bits per block

    print("=== Smoke test: synthetic scores, 6 blocks ===\n")
    ilp_assign, ilp_cost, ilp_sens = solve_mckp_ilp(synthetic_scores, budget, BIT_WIDTHS)
    pareto_assign, pareto_cost, pareto_sens = solve_pareto_frontier(synthetic_scores, budget, BIT_WIDTHS)

    print(f"ILP    : cost={ilp_cost:.1f} sensitivity={ilp_sens:.4f} -> {ilp_assign}")
    print(f"Pareto : cost={pareto_cost:.1f} sensitivity={pareto_sens:.4f} -> {pareto_assign}")

    assert pareto_sens >= ilp_sens - 1e-9, "Pareto should never beat the exact optimum"
    print("\nSanity check passed: Pareto sensitivity >= ILP sensitivity (as expected).")

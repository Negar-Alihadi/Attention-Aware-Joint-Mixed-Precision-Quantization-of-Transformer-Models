#Greedy sensitivity-per-cost bit-width allocator
import random
import itertools

BIT_WIDTHS = [2, 3, 4, 8, 16] #2, 16 is deleted for now!

def generate_synthetic_scores(block_names, seed=0):
    """
    Returns a dictionary of sensitivity scores per bit width per each block.
    """
    rng = random.Random(seed)
    scores = {}

    for name in block_names:
        fragility = rng.uniform(0.5, 3.0)
        sensitivity_by_bits = {}

        for bits in BIT_WIDTHS:
            # more bits -> less sensitivity
            sensitivity = fragility / (bits ** 1.5)
            sensitivity_by_bits[bits] = sensitivity

        scores[name] = sensitivity_by_bits

    return scores

def cost(bits, param_count=1.0):
    """
    Memory cost of storing a block at a given bit-width.
    """
    return bits * param_count

def greedy_allocate(scores, budget):
    """
    scores: dict of {block_name: {bits: sensitivity}}
    budget: max total cost allowed
      1. Give every block the HIGHEST bit-width (best accuracy, most cost).
      2. While we're over budget: find the one downgrade (one block, one
         step down in bits) that saves the most cost per unit of accuracy
         lost, and apply it. Repeat.
      3. If we still have leftover budget afterward, try upgrading blocks
         back up wherever it's affordable and helps the most.
    """
    # start at max precision
    current_bits = {name: max(BIT_WIDTHS) for name in scores}

    def total_cost():
        return sum(cost(current_bits[name]) for name in scores)

    def total_sensitivity():
        return sum(scores[name][current_bits[name]] for name in scores)

    # downgrade loop
    while total_cost() > budget:
        best_block = None
        best_new_bits = None
        best_ratio = None  # (sensitivity) / (cost) 

        for name in scores:
            bits_now = current_bits[name]
            lower_choices = [b for b in BIT_WIDTHS if b < bits_now]
            if not lower_choices:
                continue  # already at the lowest possible bit-width

            next_bits = max(lower_choices) 
            cost_saved = cost(bits_now) - cost(next_bits)
            sensitivity_added = scores[name][next_bits] - scores[name][bits_now]

            ratio = sensitivity_added / cost_saved

            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_block = name
                best_new_bits = next_bits

        if best_block is None:
            break  # can't downgrade anything further

        current_bits[best_block] = best_new_bits

    # spend remaining budget on the best upgrade available
    made_an_upgrade = True
    while made_an_upgrade:
        made_an_upgrade = False
        best_block = None
        best_new_bits = None
        best_ratio = None  # (sensitivity) / (extra cost)

        for name in scores:
            bits_now = current_bits[name]
            higher_choices = [b for b in BIT_WIDTHS if b > bits_now]
            if not higher_choices:
                continue

            next_bits = min(higher_choices)  
            extra_cost = cost(next_bits) - cost(bits_now)

            if total_cost() + extra_cost > budget:
                continue  # can't afford it

            sensitivity_saved = scores[name][bits_now] - scores[name][next_bits]
            ratio = sensitivity_saved / extra_cost

            if best_ratio is None or ratio > best_ratio:
                best_ratio = ratio
                best_block = name
                best_new_bits = next_bits

        if best_block is not None:
            current_bits[best_block] = best_new_bits
            made_an_upgrade = True

    return current_bits, total_cost(), total_sensitivity()

def brute_force_optimal(scores, budget):
    """
    Tries every possible combination of bit-widths and keeps the best one that fits the budget. Only usable for a small number of blocks.
    """
    names = list(scores.keys())
    choices_per_block = [BIT_WIDTHS] * len(names)

    best_assignment = None
    best_cost = None
    best_sensitivity = float("inf")

    for combo in itertools.product(*choices_per_block):
        total_c = sum(cost(bits) for bits in combo)
        if total_c > budget:
            continue

        total_s = sum(scores[names[i]][combo[i]] for i in range(len(names)))

        if total_s < best_sensitivity:
            best_sensitivity = total_s
            best_cost = total_c
            best_assignment = dict(zip(names, combo))

    return best_assignment, best_cost, best_sensitivity

if __name__ == "__main__":
    # 12 blocks = W_Q, W_K, W_V for 4 layers
    block_names = [f"L{layer}_{proj}" for layer in range(4) for proj in ("WQ", "WK", "WV")]
    scores = generate_synthetic_scores(block_names, seed=42)

    max_cost = sum(max(BIT_WIDTHS) for _ in block_names)
    min_cost = sum(min(BIT_WIDTHS) for _ in block_names)
    budget = min_cost + 0.35 * (max_cost - min_cost)

    print(f"{len(block_names)} blocks | min_cost={min_cost} max_cost={max_cost} budget={budget:.1f}\n")

    assignment, cost_used, sensitivity = greedy_allocate(scores, budget)
    print("Greedy allocation:")
    for name in block_names:
        print(f"  {name:10s} -> {assignment[name]:2d} bits")
    print(f"\nTotal cost: {cost_used:.1f} (budget {budget:.1f})")
    print(f"Total sensitivity: {sensitivity:.4f}")
    # validate against brute force on a small 4-block slice
    small_names = block_names[:4]
    small_scores = {name: scores[name] for name in small_names}
    small_max = sum(max(BIT_WIDTHS) for _ in small_names)
    small_min = sum(min(BIT_WIDTHS) for _ in small_names)
    small_budget = small_min + 0.35 * (small_max - small_min)

    greedy_assign, greedy_cost, greedy_sens = greedy_allocate(small_scores, small_budget)
    optimal_assign, optimal_cost, optimal_sens = brute_force_optimal(small_scores, small_budget)

    print("\n--- Validation on 4 blocks ---")
    print(f"Greedy : cost={greedy_cost:.1f} sensitivity={greedy_sens:.4f} -> {greedy_assign}")
    print(f"Optimal: cost={optimal_cost:.1f} sensitivity={optimal_sens:.4f} -> {optimal_assign}")

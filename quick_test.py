#!/usr/bin/env python3

# Quick test script to run meta-heuristic and show final allocations
import sys
sys.path.append('/home/theophile-dubois/1-Operations-Optimization-Project')

from Meta_Heuristics import MetaHeuristic
from Greedy_Algo import (
    C_ordered, C_dict, Barges, H_b, H_t_20, H_t_40,
    T_ij_list, Handling_time
)

print("Starting meta-heuristic optimization...")
print(f"Total containers: {len(C_dict)}")
print(f"Available barges: {len(Barges)}")
print(f"Barge capacities: {Barges}")

# Create and run meta-heuristic
mh = MetaHeuristic(C_ordered, C_dict, Barges,
                   H_b, H_t_20, H_t_40,
                   T_ij_list, Handling_time)

mh.initial_solution()
initial_cost, _, _ = mh.evaluate()
print(f"Initial greedy cost: €{initial_cost}")

# Run with fewer iterations for quick results
mh.local_search(max_iters=500)
print(f"Final meta-heuristic cost: €{mh.best_cost}")

# Display final allocations
mh.display_final_allocations()

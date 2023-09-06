# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:39:41 2023

@author: ragha
"""

import pandas as pd
import numpy as np

# Define the average errors and estimated costs
average_errors = {
    "MP2/ccpvdz":1.30,
    "HF/ccpvdz":2.37,
    "CCSD(T)/631g":5.88,
    "MP2/631g":6.23,
    "HF/631g":6.45,
    "CCSD(T)/sto3g":12.60,
    "MP2/sto3g":12.98,
    "HF/sto3g":15.65,
}
estimated_costs = {
    "HF/sto-3g": 1.0,
    "MP2/6-31g": 56.94241982507289,
    "CCSD(T)/cc-pvdz": 5102.040816326531
}

# Create the performance-cost matrix
performance_cost_matrix = pd.DataFrame({
    "Average Error": average_errors,
    "Estimated Cost": estimated_costs
})
print(performance_cost_matrix)

def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]

pareto_indices = identify_pareto(performance_cost_matrix.values)
pareto_front = performance_cost_matrix.iloc[pareto_indices]
print(pareto_front)

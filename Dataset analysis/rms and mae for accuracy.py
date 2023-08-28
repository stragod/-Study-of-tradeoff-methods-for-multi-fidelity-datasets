# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:20:54 2023

@author: ragha
"""

# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file and inspect it
df = pd.read_excel('C:/Users/ragha/Downloads/qm7bdata.xlsx')  # replace with your file path

# Set CCSD(T)/cc-pvdz as your benchmark
benchmark = df[(df['method'] == 'CCSD(T)') & (df['basis_set'] == 'ccpvdz')]['energy']

# Initialize lists to store results
methods = df['method'].unique()
basis_sets = df['basis_set'].unique()
results = []

# Calculate the error for each method and basis set compared to the benchmark
for method in methods:
    for basis_set in basis_sets:
        subset = df[(df['method'] == method) & (df['basis_set'] == basis_set)]['energy']
        rmse = np.sqrt(mean_squared_error(benchmark, subset))
        mae = mean_absolute_error(benchmark, subset)
        results.append({
            'Method': method,
            'Basis Set': basis_set,
            'RMSE': rmse,
            'MAE': mae
        })

# Convert the results into a DataFrame
results_df = pd.DataFrame(results)

# Display the sorted DataFrame
display(results_df.sort_values(['RMSE', 'MAE']))

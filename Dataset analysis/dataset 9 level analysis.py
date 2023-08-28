# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 21:59:12 2023

@author: ragha
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load the data
with open('C:/Users/ragha/OneDrive - SRH IT/Documents/thesis/molecule_data/qm7b.json', 'r') as f:
    data = json.load(f)

# Initialize lists to store differences
differences_mp2_sto3g = []
differences_mp2_631g = []
differences_mp2_ccpvdz = []
differences_CCSDT_631g = []
differences_CCSDT_sto3g = []

# Loop through all the molecules and calculate differences
for molecule_id, target in data["targets"].items():
    benchmark_energy = target["CCSD(T)"]["ccpvdz"]
    differences_mp2_sto3g.append(target["MP2"]["sto3g"] - benchmark_energy)
    differences_mp2_631g.append(target["MP2"]["631g"] - benchmark_energy)
    differences_mp2_ccpvdz.append(target["MP2"]["ccpvdz"] - benchmark_energy)
    differences_CCSDT_631g.append(target["CCSD(T)"]["631g"] - benchmark_energy)
    differences_CCSDT_sto3g.append(target["CCSD(T)"]["sto3g"] - benchmark_energy)

# Function to plot cumulative distribution
def plot_cdf(data, label):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
    plt.plot(sorted_data, yvals, label=label)

# Plot cumulative distributions
plot_cdf(differences_mp2_sto3g, 'MP2/sto-3g')
plot_cdf(differences_mp2_631g, 'MP2/6-31g')
plot_cdf(differences_mp2_ccpvdz, 'MP2/cc-pvdz')
plot_cdf(differences_CCSDT_631g, 'CCSD(T)/6-31g')
plot_cdf(differences_CCSDT_sto3g, 'CCSD(T)/sto-3g')

plt.title('Cumulative Distribution of Differences from CCSD(T)/cc-pVDZ')
plt.xlabel('Energy Difference (kcal/mol)')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid()
plt.show()

# Function to calculate and print statistics
def print_statistics(data, label):
    mean_difference = np.mean(data)
    median_difference = np.median(data)
    std_difference = np.std(data)
    percentile_25 = np.percentile(data, 25)
    percentile_75 = np.percentile(data, 75)
    
    print(f'Statistics for {label}:')
    print(f'Mean Difference: {mean_difference:.2f} kcal/mol')
    print(f'Median Difference: {median_difference:.2f} kcal/mol')
    print(f'Standard Deviation: {std_difference:.2f} kcal/mol')
    print(f'25th Percentile: {percentile_25:.2f} kcal/mol')
    print(f'75th Percentile: {percentile_75:.2f} kcal/mol')
    print('--------------------------------------')

# Print statistics for each combination
print_statistics(differences_mp2_sto3g, 'MP2/sto-3g')
print_statistics(differences_mp2_631g, 'MP2/6-31g')
print_statistics(differences_mp2_ccpvdz, 'MP2/cc-pvdz')
print_statistics(differences_CCSDT_631g, 'CCSD(T)/6-31g')
print_statistics(differences_CCSDT_sto3g, 'CCSD(T)/sto-3g')

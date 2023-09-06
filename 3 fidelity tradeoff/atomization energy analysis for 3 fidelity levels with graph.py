# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:13:26 2023

@author: ragha
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset
with open('C:/Users/ragha/OneDrive - SRH IT/Documents/thesis/molecule_data/qm7b.json', 'r') as f:
    data = json.load(f)

lowest_fidelity = []
intermediate_fidelity = []
highest_fidelity = []
true_fidelity = []

for molecule_id, target in data['targets'].items():
    lowest_fidelity.append(target['HF']['sto3g'])
    intermediate_fidelity.append(target['MP2']['631g'])
    highest_fidelity.append(target['MP2']['ccpvdz'])
    true_fidelity.append(target['CCSD(T)']['ccpvdz'])

fidelity_df = pd.DataFrame({'Lowest Fidelity': lowest_fidelity,
                            'Intermediate Fidelity': intermediate_fidelity,
                            'Highest Fidelity': highest_fidelity,'true value':true_fidelity})

lowest_fidelity_errors = np.abs(np.array(lowest_fidelity) - np.array(true_fidelity))
intermediate_fidelity_errors = np.abs(np.array(intermediate_fidelity) - np.array(true_fidelity))
high_fidelity_errors = np.abs(np.array(highest_fidelity) - np.array(true_fidelity))
lowest_fidelity_mean_error = np.mean(lowest_fidelity_errors)
lowest_fidelity_std_error = np.std(lowest_fidelity_errors)

intermediate_fidelity_mean_error = np.mean(intermediate_fidelity_errors)
intermediate_fidelity_std_error = np.std(intermediate_fidelity_errors)

high_fidelity_mean_error = np.mean(high_fidelity_errors)
high_fidelity_std_error = np.std(high_fidelity_errors)

print("Lowest fidelity mean error:", lowest_fidelity_mean_error)
print("Lowest fidelity standard deviation of error:", lowest_fidelity_std_error)
print("Intermediate fidelity mean error:", intermediate_fidelity_mean_error)
print("Intermediate fidelity standard deviation of error:", intermediate_fidelity_std_error)
print("high fidelity mean error:", high_fidelity_mean_error)
print("high fidelity standard deviation of error:", high_fidelity_std_error)

print(fidelity_df.describe())
fidelity_df.hist(bins=50, figsize=(15, 7))
plt.show()
plt.figure(figsize=(10, 5))
sns.boxplot(data=fidelity_df)
plt.show()
sns.pairplot(fidelity_df)
plt.show()
correlations = fidelity_df.corr()
print(correlations)
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.show()


# Assuming highest_fidelity data is the reference
reference_data = true_fidelity

lowest_fidelity_errors = [abs(low - ref) for low, ref in zip(lowest_fidelity, reference_data)]
intermediate_fidelity_errors = [abs(inter - ref) for inter, ref in zip(intermediate_fidelity, reference_data)]
high_fidelity_errors = [abs(inter - ref) for inter, ref in zip(highest_fidelity, reference_data)]
# create a dataframe for all fidelity levels
error_grid_df = pd.DataFrame({'HF_sto3g': lowest_fidelity_errors,
                              'MP2_631g': intermediate_fidelity_errors,
                              'MP2_cc-pvdz': highest_fidelity})

#average error for each combination
average_errors = error_grid_df.mean()

#create a 3x3 dataframe to visuzilse the error grid
error_grid = pd.DataFrame({'sto3g': [average_errors['HF_sto3g'], None, None],
                           '6-31g': [None, average_errors['MP2_631g'], None],
                           'cc-pvdz': [None, None, average_errors['MP2_cc-pvdz']]},
                          index=['HF', 'MP2', 'CCSD(T)'])


# heatmap to visualize the error grid
plt.figure(figsize=(7, 6))
sns.heatmap(error_grid, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.xlabel('Basis Set')
plt.ylabel('Method')
plt.title('Average Errors for Different Fidelity Levels')
plt.show()



# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:24:36 2023

@author: ragha
"""

# Load the dataset from an excel file


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
# Load the dataset from an excel file
df = pd.read_excel('C:/Users/ragha/Downloads/qm7bdata.xlsx')

# Pivot the table
pivot_df = df.pivot(index='molecule_id', columns='fidelity_level', values='energy')

# Compute the correlation matrix
correlation_matrix = pivot_df.corr()

# Create a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", square=True, cmap = 'Blues')

plt.title("Correlation Heatmap")
plt.show()

print(correlation_matrix)

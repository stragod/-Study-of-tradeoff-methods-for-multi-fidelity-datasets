# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 08:11:55 2023

@author: ragha
"""

import pandas as pd
import numpy as np
import json

# Load the data into pandas DataFrames
with open("C:/Users/ragha/Downloads/molecule_data/G4MP2.json", "r") as f:
    data = json.load(f)

g4mp2_df = pd.DataFrame(data["G4MP2"]["U0"].items(), columns=["ID", "G4MP2_energy"]).set_index("ID")
b3lyp_df = pd.DataFrame(data["B3LYP"]["U0"].items(), columns=["ID", "B3LYP_energy"]).set_index("ID")

g4mp2_mol_df = pd.DataFrame(data["G4MP2"]["molecules"]).T
b3lyp_mol_df = pd.DataFrame(data["B3LYP"]["molecules"]).T

g4mp2_df = g4mp2_df.merge(g4mp2_mol_df, left_index=True, right_index=True)
b3lyp_df = b3lyp_df.merge(b3lyp_mol_df, left_index=True, right_index=True)

# Merge the two DataFrames based on the molecule ID
merged_df = g4mp2_df.merge(b3lyp_df, left_index=True, right_index=True, suffixes=('_g4mp2', '_b3lyp'))

# Add the number of sites (atoms) in the molecules
merged_df["num_sites_g4mp2"] = merged_df["sites_g4mp2"].apply(len)
merged_df["num_sites_b3lyp"] = merged_df["sites_b3lyp"].apply(len)

# Check for any missing or infinite values in the data
print("Missing values in G4MP2_energy:", merged_df["G4MP2_energy"].isnull().sum())
print("Missing values in B3LYP_energy:", merged_df["B3LYP_energy"].isnull().sum())

# Remove any rows with missing or infinite values
merged_df_clean = merged_df.dropna(subset=["G4MP2_energy", "B3LYP_energy"])
merged_df_clean = merged_df_clean[np.isfinite(merged_df_clean["G4MP2_energy"]) & np.isfinite(merged_df_clean["B3LYP_energy"])]


# Calculate the correlation coefficients using Pandas
corr_g4mp2 = merged_df_clean['num_sites_g4mp2'].corr(merged_df_clean['G4MP2_energy'])
corr_b3lyp = merged_df_clean['num_sites_b3lyp'].corr(merged_df_clean['B3LYP_energy'])

print("Correlation coefficient for G4MP2 (high-fidelity) data:", corr_g4mp2)
print("Correlation coefficient for B3LYP (low-fidelity) data:", corr_b3lyp)

# Calculate the correlation coefficients between the number of sites and energy values for both high fidelity and low fidelity data
#corr_g4mp2 = np.corrcoef(merged_df_clean["num_sites_g4mp2"], merged_df_clean["G4MP2_energy"])[0, 1]
#corr_b3lyp = np.corrcoef(merged_df_clean["num_sites_b3lyp"], merged_df_clean["B3LYP_energy"])[0, 1]

#print("Correlation coefficient for G4MP2 (high-fidelity) data:", corr_g4mp2)
#print("Correlation coefficient for B3LYP (low-fidelity) data:", corr_b3lyp)

import matplotlib.pyplot as plt

plt.hist(merged_df_clean['G4MP2_energy'], bins=50)
plt.xlabel('G4MP2 Energy (eV)')
plt.ylabel('Frequency')
plt.title('Distribution of G4MP2 Energy Values')
plt.show()

plt.hist(merged_df_clean['num_sites_g4mp2'], bins=50)
plt.xlabel('Number of Sites')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Sites in G4MP2 Data')
plt.show()

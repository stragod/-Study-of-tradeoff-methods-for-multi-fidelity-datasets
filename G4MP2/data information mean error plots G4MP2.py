# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:48:41 2023

@author: ragha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Calculate the error between G4MP2 and B3LYP energies
# Calculate the error between G4MP2 and B3LYP energies
merged_df["energy_error"] = np.abs(merged_df["G4MP2_energy"] - merged_df["B3LYP_energy"])

# Bin the molecules based on the number of sites
num_bins = 10
merged_df["num_sites_bin"] = pd.cut(merged_df["num_sites_g4mp2"], bins=num_bins, labels=False)

# Group the data by the binned number of sites and calculate the mean error for each group
#mean_error_by_sites = merged_df.groupby("num_sites_bin")["energy_error"].mean()

# Plot the mean error as a function of the number of sites
#plt.plot(mean_error_by_sites.index, mean_error_by_sites.values, marker='o')
#plt.xlabel("Binned Number of Sites")
#plt.ylabel("Mean Energy Error (eV)")
#plt.title("Mean Energy Error vs Binned Number of Sites")
#plt.show()




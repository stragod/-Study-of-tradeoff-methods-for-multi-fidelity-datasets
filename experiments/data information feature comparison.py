# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:27:05 2023

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

# Investigate the relationship between the charge and the energies
plt.scatter(merged_df["charge_g4mp2"], merged_df["G4MP2_energy"], alpha=0.5, label="G4MP2")
plt.scatter(merged_df["charge_b3lyp"], merged_df["B3LYP_energy"], alpha=0.5, label="B3LYP")
plt.xlabel("Charge")
plt.ylabel("Energy (eV)")
plt.title("Charge vs Energy for G4MP2 and B3LYP Methods")
plt.legend()
plt.show()

# Investigate the relationship between the spin multiplicity and the energies
plt.scatter(merged_df["spin_multiplicity_g4mp2"], merged_df["G4MP2_energy"], alpha=0.5, label="G4MP2")
plt.scatter(merged_df["spin_multiplicity_b3lyp"], merged_df["B3LYP_energy"], alpha=0.5, label="B3LYP")
plt.xlabel("Spin Multiplicity")
plt.ylabel("Energy (eV)")
plt.title("Spin Multiplicity vs Energy for G4MP2 and B3LYP Methods")
plt.legend()
plt.show()




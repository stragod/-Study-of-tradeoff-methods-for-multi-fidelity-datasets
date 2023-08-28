import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

with open("C:/Users/ragha/Downloads/molecule_data/G4MP2.json", "r") as f:
    data = json.load(f)
    
g4mp2_df = pd.DataFrame(data["G4MP2"]["U0"].items(), columns=["ID", "G4MP2_energy"]).set_index("ID")
b3lyp_df = pd.DataFrame(data["B3LYP"]["U0"].items(), columns=["ID", "B3LYP_energy"]).set_index("ID")

g4mp2_df.columns = ["G4MP2_energy"]
b3lyp_df.columns = ["B3LYP_energy"]
merged_df = g4mp2_df.merge(b3lyp_df, left_index=True, right_index=True)
plt.scatter(merged_df["G4MP2_energy"], merged_df["B3LYP_energy"], alpha=0.5)
plt.xlabel("G4MP2 Energy (eV)")
plt.ylabel("B3LYP Energy (eV)")
plt.title("G4MP2 vs B3LYP Energies for QM9 Molecules")
plt.show()
print("Mean G4MP2 energy:", merged_df["G4MP2_energy"].mean())
print("Standard deviation of G4MP2 energy:", merged_df["G4MP2_energy"].std())

print("Mean B3LYP energy:", merged_df["B3LYP_energy"].mean())
print("Standard deviation of B3LYP energy:", merged_df["B3LYP_energy"].std())
correlation = merged_df["G4MP2_energy"].corr(merged_df["B3LYP_energy"])
print("Correlation between G4MP2 and B3LYP energies:", correlation)

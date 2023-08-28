# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:38:19 2023

@author: ragha
"""

import json
import pandas as pd

# Load the dataset
with open('C:/Users/ragha/Downloads/molecule_data/G4MP2.json', 'r') as f:
    data = json.load(f)

# Extract the G4MP2 and B3LYP data
g4mp2_data = data['G4MP2']['molecules']
b3lyp_data = data['B3LYP']['molecules']

# Extract and label the molecule features
features = {}
for i, molecule in g4mp2_data.items():
    for prop in molecule.keys():
        features[prop] = True

# Create a dataframe for the G4MP2 data
g4mp2_df = pd.DataFrame(columns=list(features.keys()))
for i, molecule in g4mp2_data.items():
    row = []
    for prop in features.keys():
        if prop in molecule:
            row.append(molecule[prop])
        else:
            row.append(None)
    g4mp2_df.loc[i] = row

# Create a dataframe for the B3LYP data
b3lyp_df = pd.DataFrame(columns=list(features.keys()))
for i, molecule in b3lyp_data.items():
    row = []
    for prop in features.keys():
        if prop in molecule:
            row.append(molecule[prop])
        else:
            row.append(None)
    b3lyp_df.loc[i] = row

# Print the head of the G4MP2 dataframe
print('G4MP2 Dataframe:\n')
print(g4mp2_df.head())

# Print the head of the B3LYP dataframe
print('\nB3LYP Dataframe:\n')
print(b3lyp_df.head())
# Print info about the G4MP2 dataframe
print('G4MP2 Dataframe Info:\n')
print(g4mp2_df.info())

# Print info about the B3LYP dataframe
print('\nB3LYP Dataframe Info:\n')
print(b3lyp_df.info())

print(g4mp2_df.describe())
print(b3lyp_df.describe())
print(g4mp2_df.describe())
print(b3lyp_df.describe())

"""
Created on Mon Apr 24 14:22:09 2023

@author: ragha
"""
import json

# Load the JSON data into a dictionary
with open('C:/Users/ragha/Downloads/molecule_data/G4MP2.json', 'r') as f:
    data = json.load(f)

print(data.keys())

print(data['G4MP2']['U0'].values())[0]

# Print the IDs of the first 5 molecules
print(list(data['G4MP2']['U0'].keys())[:5])

# Print the molecular structure of the first molecule
#print(data['G4MP2']['molecules']['mol-1'])
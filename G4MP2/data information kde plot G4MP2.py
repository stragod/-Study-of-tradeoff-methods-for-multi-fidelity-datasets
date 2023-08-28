import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymatgen.core import Molecule

def extract_features(mol_dict):
    mol = Molecule.from_dict(mol_dict)
    num_atoms = len(mol)
    num_bonds = len(mol.get_covalent_bonds())
    element_counts = mol.composition.get_el_amt_dict()
    
    # Combine the features into a single list
    features = [num_atoms, num_bonds] + list(element_counts.values())
    return features

def load_data(G4MP2):
    with open('C:/Users/ragha/Downloads/molecule_data/G4MP2.json', 'r') as f:
        data = json.load(f)
    return data

def prepare_data(data):
    high_fidelity = data['G4MP2']['U0']
    low_fidelity = data['B3LYP']['U0']
    
    high_fidelity_data = [(extract_features(data['G4MP2']['molecules'][str(idx)]), energy) for idx, energy in high_fidelity.items()]
    low_fidelity_data = [(extract_features(data['B3LYP']['molecules'][str(idx)]), energy) for idx, energy in low_fidelity.items()]
    
    return high_fidelity_data, low_fidelity_data

filename = "path/to/G4MP2.json"
data = load_data(filename)
high_fidelity_data, low_fidelity_data = prepare_data(data)

# Extract the feature values for high and low fidelity data
high_fidelity_features = [item[0] for item in high_fidelity_data]
low_fidelity_features = [item[0] for item in low_fidelity_data]

# Create a DataFrame containing the feature values and fidelity types
high_fidelity_df = pd.DataFrame(high_fidelity_features)
high_fidelity_df['Fidelity'] = 'High'
low_fidelity_df = pd.DataFrame(low_fidelity_features)
low_fidelity_df['Fidelity'] = 'Low'

data_df = pd.concat([high_fidelity_df, low_fidelity_df])

# Convert high_fidelity_features and low_fidelity_features to DataFrames
high_df = pd.DataFrame(high_fidelity_features, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5','Feature6','Feature7'])
high_df['Fidelity'] = 'High'
low_df = pd.DataFrame(low_fidelity_features, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5','Feature6','Feature7'])
low_df['Fidelity'] = 'Low'

# Combine high and low fidelity DataFrames
data_df = pd.concat([high_df, low_df])

# Create separate KDE plots for each feature
for i in range(1, 8):
    sns.kdeplot(data=data_df, x=f'Feature{i}', hue='Fidelity', common_norm=False, fill=True, bw_adjust=3, alpha=0.5)
    plt.title(f'Feature {i} Distribution')
    plt.show()
    # Create a KDE plot
#sns.kdeplot(data=data_df, x=0, hue='Fidelity', common_norm=False, fill=True, bw_adjust=1,alpha=0.5)
#print("High fidelity features:", high_fidelity_features)
#print("Low fidelity features:", low_fidelity_features)

# Set labels and legend
#plt.xlabel('Number of Atoms')
#plt.ylabel('Density')

# Show the plot
#plt.show()

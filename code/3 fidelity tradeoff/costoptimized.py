# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:07:53 2023

@author: ragha
"""

import json
import numpy as np
from pymatgen.core import Molecule

def estimate_num_basis_functions(mol_dict, basis):
    # Define the number of basis functions for each atom in each basis set
    basis_functions = {
        "H": {"sto-3g": 1, "6-31g": 4, "cc-pvdz": 6},
        "C": {"sto-3g": 5, "6-31g": 13, "cc-pvdz": 25},
        "N": {"sto-3g": 5,"6-31g": 13,"cc-pvdz": 25},
        "O": {"sto-3g": 5,"6-31g": 13,"cc-pvdz": 25},
        "S": {"sto-3g": 9,"6-31g": 22,"cc-pvdz": 55},
        "Cl": {"sto-3g": 9,"6-31g": 22,"cc-pvdz": 55}
        }

    mol = Molecule.from_dict(mol_dict)
    num_basis_functions = 0
    for atom in mol:
        atom_type = str(atom.specie)
        if atom_type in basis_functions:
            num_basis_functions += basis_functions[atom_type][basis]
        else:
            raise ValueError(f"Unknown atom type: {atom_type}")

    return num_basis_functions


def estimate_cost(num_basis_functions, method):
    if method == "HF":
        return num_basis_functions ** 4
    elif method == "MP2":
        return num_basis_functions ** 5
    elif method == "CCSD(T)":
        return num_basis_functions ** 7
    else:
        raise ValueError(f"Unknown method: {method}")

with open("C:/Users/ragha/OneDrive - SRH IT/Documents/thesis/molecule_data/qm7b.json", "r") as f:
    data = json.load(f)

molecules = data["molecules"]
sample_size = 7211  # Adjust this number based on your needs and available resources
sample_molecules = np.random.choice(list(molecules.keys()), size=sample_size, replace=False)

methods = ["HF", "MP2", "CCSD(T)"]
basis_sets = ["sto-3g", "6-31g", "cc-pvdz"]
results = {}

for method in methods:
    for basis in basis_sets:
        key = f"{method}/{basis}"
        results[key] = {"estimated_costs": []}

        for mol_id in sample_molecules:
            mol_dict = molecules[mol_id]
            num_basis_functions = estimate_num_basis_functions(mol_dict, basis)
            estimated_cost = estimate_cost(num_basis_functions, method)
            results[key]["estimated_costs"].append(estimated_cost)

# Define scaling factors for each method and basis set
method_scaling = {"HF": 4, "MP2": 5, "CCSD(T)": 7}
basis_scaling = {"sto-3g": 7, "6-31g": 25, "cc-pvdz": 100}

for key, value in results.items():
    # Calculate the estimated cost
    method, basis = key.split("/")
    scaling_factor = method_scaling[method]
    n_basis = basis_scaling[basis]
    estimated_cost = scaling_factor * (n_basis ** 3)  # Using N^3 scaling as a rough estimate
    value['estimated_cost'] = estimated_cost

# find the minimum estimated cost
min_cost = min(results.values(), key=lambda x: x['estimated_cost'])['estimated_cost']

# Normalize the estimated costs
for key, value in results.items():
    normalized_cost = value['estimated_cost'] / min_cost
    print(f"{key}: Normalized estimated cost = {normalized_cost}")

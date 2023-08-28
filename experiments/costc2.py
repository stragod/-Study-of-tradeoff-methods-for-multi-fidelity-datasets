# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:25:10 2023

@author: ragha
"""

import json
import psi4
from pymatgen.core import Molecule

def get_basis_functions(molecule_dict, basis_set):
    molecule = Molecule.from_dict(molecule_dict)
    psi4_molecule = psi4.geometry("\n".join([f"{atom.species} {atom.x} {atom.y} {atom.z}" for atom in molecule]))
    
    basis = psi4.core.BasisSet.build(psi4_molecule, "BASIS", basis_set)
    return basis.nbf()

def estimate_cost(molecule_dict, method, basis_set):
    n = get_basis_functions(molecule_dict, basis_set)
    scaling_factors = {"HF": 4, "MP2": 5, "CCSD(T)": 7}
    cost = n ** scaling_factors[method]
    return cost

# Load the data from the qm7b.json file
with open("qm7b.json", "r") as f:
    data = json.load(f)

molecules = data["molecules"]
targets = data["targets"]

# Basis set names in psi4 format
basis_sets_psi4 = {"sto3g": "sto-3g", "631g": "6-31g", "cc-pvdz": "cc-pvdz"}

# Estimate the computational cost for each molecule, method, and basis set combination
costs = {}
for molecule_id, molecule_dict in molecules.items():
    costs[molecule_id] = {}
    for method in targets[molecule_id].keys():
        costs[molecule_id][method] = {}
        for basis_set in targets[molecule_id][method].keys():
            costs[molecule_id][method][basis_set] = estimate_cost(molecule_dict, method, basis_sets_psi4[basis_set])

# Print the results
for molecule_id, molecule_costs in costs.items():
    print(f"Costs for molecule {molecule_id}:")
    for method, basis_set_costs in molecule_costs.items():
        print(f"  {method}:")
        for basis_set, cost in basis_set_costs.items():
            print(f"    {basis_set}: {cost}")

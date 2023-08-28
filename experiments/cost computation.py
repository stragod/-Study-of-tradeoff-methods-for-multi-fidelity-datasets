# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:07:51 2023

@author: ragha
"""
"""
import json
from pymatgen.core import Molecule
from pyscf import gto

def get_basis_functions(molecule_dict, basis_set):
    molecule = Molecule.from_dict(molecule_dict)
    atoms = " ".join([f"{atom.species} {atom.x} {atom.y} {atom.z}" for atom in molecule])
    
    mol = gto.M(atom=atoms, basis=basis_set)
    return mol.nao

def estimate_cost(molecule_dict, method, basis_set):
    n = get_basis_functions(molecule_dict, basis_set)
    scaling_factors = {"HF": 4, "MP2": 5, "CCSD(T)": 7}
    cost = n ** scaling_factors[method]
    return cost

# Load the data from the qm7b.json file
with open("/mnt/c/Users/ragha/Downloads/qm7b.json", "r") as f:
    data = json.load(f)

molecules = data["molecules"]
targets = data["targets"]

# Basis set names in pyscf format
basis_sets_pyscf = {"sto3g": "sto-3g", "631g": "6-31g", "cc-pvdz": "cc-pvdz"}

# Estimate the computational cost for each molecule, method, and basis set combination
costs = {}
for molecule_id, molecule_dict in molecules.items():
    costs[molecule_id] = {}
    for method in targets[molecule_id].keys():
        costs[molecule_id][method] = {}
        for basis_set in targets[molecule_id][method].keys():
            costs[molecule_id][method][basis_set] = estimate_cost(molecule_dict, method, basis_sets_pyscf[basis_set])

# Print the results
for molecule_id, molecule_costs in costs.items():
    print(f"Costs for molecule {molecule_id}:")
    for method, basis_set_costs in molecule_costs.items():
        print(f"  {method}:")
        for basis_set, cost in basis_set_costs.items():
            print(f"    {basis_set}: {cost}")
            
            
    psi4.geometry(molecule)
    energy = psi4.energy(f"{method}/{basis}")

    return energy
"""
import psi4
import json
import numpy as np
import time
with open("/mnt/c/Users/ragha/Downloads/qm7b.json", "r") as f:
    data = json.load(f)

molecules = data["molecules"]
sample_size = 100  # Adjust this number based on your needs and available resources
sample_molecules = np.random.choice(list(molecules.values()), size=sample_size, replace=False)
def run_calculation(molecule, method, basis):
    psi4.set_memory("2 GB")
    psi4.set_num_threads(2)


methods = ["HF", "MP2", "CCSD(T)"]
basis_sets = ["sto-3g", "6-31g", "cc-pvdz"]
results = {}

for method in methods:
    for basis in basis_sets:
        key = f"{method}/{basis}"
        results[key] = {"cpu_time": [], "energies": []}

        for mol in sample_molecules:
            start_time = time.time()
            energy = run_calculation(mol, method, basis)
            end_time = time.time()

            cpu_time = end_time - start_time
            results[key]["cpu_time"].append(cpu_time)
            results[key]["energies"].append(energy)

# Calculate average CPU time and memory usage (if applicable) for each method and basis set combination
for key, value in results.items():
    avg_cpu_time = np.mean(value["cpu_time"])
    # avg_memory = ...  # Calculate average memory usage if you want to include it in your analysis
    print(f"{key}: Average CPU time = {avg_cpu_time} seconds")




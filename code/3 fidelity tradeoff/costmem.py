import json
import numpy as np
from pymatgen.core import Molecule

def estimate_num_basis_functions(mol_dict, basis):
    mol = Molecule.from_dict(mol_dict)
    num_atoms = len(mol)
    if basis == "sto-3g":
        return num_atoms * 3
    elif basis == "6-31g":
        return num_atoms * 6
    elif basis == "cc-pvdz":
        return num_atoms * 10
    else:
        raise ValueError(f"Unknown basis set: {basis}")

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
sample_size = 1500  # Adjust this number based on your needs and available resources
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

# Now, find the minimum estimated cost
min_cost = min(results.values(), key=lambda x: x['estimated_cost'])['estimated_cost']

# Normalize the estimated costs
for key, value in results.items():
    normalized_cost = value['estimated_cost'] / min_cost
    print(f"{key}: Normalized estimated cost = {normalized_cost}")

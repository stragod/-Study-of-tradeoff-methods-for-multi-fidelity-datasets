import json
import numpy as np
import psi4

def molecule_from_dict(molecule_dict):
    atom_list = [f"{atom['species'][0]['element']} {atom['xyz'][0]} {atom['xyz'][1]} {atom['xyz'][2]}" for atom in molecule_dict["sites"]]
    molecule_str = "\n".join(atom_list)
    return psi4.geometry(molecule_str)

def basis_function_count(molecule, basis_set):
    for atom in range(molecule.natom()):
        symbol = molecule.symbol(atom)
        label = molecule.label(atom)
        molecule.set_basis_by_symbol(label, symbol, basis_set)
    basis = psi4.core.BasisSet.build(molecule)
    return basis.nbf()

def estimate_cost(molecule, method, basis_set):
    n = basis_function_count(molecule, basis_set)
    scaling_factors = {"HF": 4, "MP2": 5, "CCSD(T)": 7}
    cost = n ** scaling_factors[method]
    return cost

with open("qm7b.json", "r") as f:
    data = json.load(f)

molecules = data["molecules"]
targets = data["targets"]

costs = {}
for molecule_id, molecule_dict in molecules.items():
    molecule = molecule_from_dict(molecule_dict)
    costs[molecule_id] = {}
    for method in targets[molecule_id].keys():
        costs[molecule_id][method] = {}
        for basis_set in targets[molecule_id][method].keys():
            costs[molecule_id][method][basis_set] = estimate_cost(molecule, method, basis_set)

for molecule_id, molecule_costs in costs.items():
    print(f"Costs for molecule {molecule_id}:")
    for method, basis_set_costs in molecule_costs.items():
        print(f"  {method}:")
        for basis_set, cost in basis_set_costs.items():
            print(f"    {basis_set}: {cost}")

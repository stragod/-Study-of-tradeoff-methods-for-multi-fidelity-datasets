# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:57:17 2023

@author: ragha
"""

import json
import time
import psi4
import numpy as np
from pymatgen.core import Molecule

def run_calculation(molecule_dict, method, basis):
    # Create a Psi4 geometry from a Pymatgen Molecule object
    mol = Molecule.from_dict(molecule_dict)
    atoms = "\n".join([f"{atom.species} {atom.x} {atom.y} {atom.z}" for atom in mol])
    psi4.geometry(atoms)

    # Set the computation options
    psi4.set_options({"basis": basis})

    # Run the energy calculation
    energy = psi4.energy(method)

    return energy

def main():
    methods = ["HF", "MP2", "CCSD(T)"]
    bases = ["sto-3g", "6-31g", "cc-pvdz"]

    with open('/mnt/c/Users/ragha/Downloads/qm7b.json', 'r') as f:
        data = json.load(f)

    molecules = data["molecules"]
    targets = data["targets"]

    results = {}

    for method in methods:
        for basis in bases:
            start_times = []
            for ID, molecule_dict in molecules.items():
                start = time.time()
                energy = run_calculation(molecule_dict, method, basis)
                end = time.time()

                # Compute the difference with the target value
                target = targets[ID][method][basis]
                difference = energy - target

                results.setdefault(method, {}).setdefault(basis, []).append(difference)

                start_times.append(end-start)

            avg_cpu_time = np.mean(start_times)
            print(f"{method}/{basis}: Average CPU time = {avg_cpu_time} seconds")

if __name__ == "__main__":
    main()



# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:02:31 2023

@author: ragha
"""
import argeparse
import json
from itertools import product

from pyprojroot import here

import mf2

from experiments import Instance, create_model_error_grid

# Load the data from the JSON file
with open('C:/Users/ragha/Downloads/molecule_data/G4MP2.json') as f:
    data = json.load(f)

# Extract the energies and molecules for the G4MP2 and B3LYP calculations
g4mp2_energies = data["G4MP2"]["U0"]
g4mp2_molecules = data["G4MP2"]["molecules"]
b3lyp_energies = data["B3LYP"]["U0"]
b3lyp_molecules = data["B3LYP"]["molecules"]

save_dir = here('C:/Users/ragha/Downloads/model_error_grid/')
save_dir.mkdir(parents=True, exist_ok=True)

# Use the molecule IDs from your data instead of generating new ones


kernels = ['Matern']
scaling_options = [
    'off',
    # 'on',
    # 'inverted',
    # 'regularized'
]

min_high, max_high = 2, 24
min_low, max_low = 3, 60
extra_attributes = {'mf2_version': mf2.__version__}
def main(args):
   case_ids = list(g4mp2_energies.keys()) 
if args.idx:
     case_ids = [case_ids[idx] for idx in args.idx]
instances = [Instance(h, l, r)
             for h, l, r in product(range(min_high, max_high + 1),
                                    range(min_low, max_low + 1),
                                    range(args.numreps))
             if h < l]

for case_id in case_ids:
    # Define a function that returns the energies and molecules for a specific molecule ID
    def get_energies_and_molecules(molecule_id):
        return (g4mp2_energies[molecule_id], b3lyp_energies[molecule_id]), (g4mp2_molecules[molecule_id], b3lyp_molecules[molecule_id])

    for kernel, scale in product(kernels, scaling_options):
        # Call create_model_error_grid() with the get_energies_and_molecules() function
        create_model_error_grid(get_energies_and_molecules, kernel, scale, instances, save_dir=save_dir,
                                extra_attributes=extra_attributes)

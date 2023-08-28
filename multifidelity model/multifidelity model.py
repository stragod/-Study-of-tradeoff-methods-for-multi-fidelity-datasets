# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:02:06 2023

@author: ragha
"""

import json
from pymatgen.core import Molecule
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Function to calculate Coulomb matrix
def coulomb_matrix(mol):
    num_atoms = len(mol)
    mat = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(i+1):
            if i == j:
                mat[i, i] = 0.5 * mol[i].specie.Z ** 2.4
            else:
                mat[i, j] = mol[i].specie.Z * mol[j].specie.Z / mol.get_distance(i, j)
                mat[j, i] = mat[i, j]
    return mat

# Function to sort Coulomb matrix
def sort_coulomb_matrix(mat):
    row_norms = np.linalg.norm(mat, axis=1)
    sorted_indices = np.argsort(row_norms)
    return mat[sorted_indices, :][:, sorted_indices]

# Load data from JSON
with open('C:/Users/ragha/OneDrive - SRH IT/Documents/thesis/molecule_data/qm7b.json', 'r') as f:
    data = json.load(f)

# Convert data to pymatgen Molecule objects
molecules = {ID: Molecule.from_dict(mol_dict) for ID, mol_dict in data['molecules'].items()}

# Generate Coulomb matrices
coulomb_matrices = {ID: coulomb_matrix(mol) for ID, mol in molecules.items()}

# Pad Coulomb matrices to same size
max_num_atoms = max(len(mol) for mol in molecules.values())
padded_matrices = {ID: np.pad(mat, ((0, max_num_atoms - len(mat)), (0, max_num_atoms - len(mat))), constant_values=0) 
                   for ID, mat in coulomb_matrices.items()}

# Sort Coulomb matrices
sorted_matrices = {ID: sort_coulomb_matrix(mat) for ID, mat in padded_matrices.items()}

sorted_matrices = {ID: sort_coulomb_matrix(mat) for ID, mat in padded_matrices.items()}

# First, we need to ensure that our sorted matrices are in an array, not a dictionary.
# And also we need to make sure that the size of the coulomb matrices matches with the NUM_ATOMS variable

# Convert sorted matrices to a list and resize to match NUM_ATOMS
NUM_ATOMS = 23
sorted_coulomb_list = [np.pad(mat, ((0, NUM_ATOMS - len(mat)), (0, NUM_ATOMS - len(mat))), constant_values=0) 
                       for mat in sorted_matrices.values()]
sorted_coulomb_array = np.array(sorted_coulomb_list)

# X is set of Coulomb matrices
X = sorted_coulomb_array
X_reshaped = X.reshape(len(sorted_matrices), NUM_ATOMS*NUM_ATOMS)

# y is their ground-truth atomization energies, for simplicity let's take the CCSD(T) method with cc-pvdz base
y = np.array([val["CCSD(T)"]["ccpvdz"] for val in data["targets"].values()])
y_scaling_factor = np.max(np.absolute(y))
y_scaled = y/y_scaling_factor

# Constants
NUM_SAMPLES = len(sorted_matrices)
RANDOM_SEED = 42


# List of keys (molecule IDs)
ids = list(sorted_matrices.keys())

eigval_X = []
eigvec_X = []
for i in range(X.shape[0]):
    eigval, eigvec = np.linalg.eig(X[i, :, :])
    eigval_X.append(eigval)
    eigvec_X.append(eigvec.flatten())

# reshape eigval_X into 2D array with X.shape[0] rows
eigval_X_reshaped = np.array(eigval_X).reshape(X.shape[0], -1)

X_reshaped_with_eigfeature = np.concatenate((np.array(eigvec_X), eigval_X_reshaped, X_reshaped), axis=1).astype(np.float64)
print("Dimensions of X_reshaped_with_eigfeature: {}".format(X_reshaped_with_eigfeature.shape))


# Compute eigenvector centralities
eigcentralities = []

for matrix in sorted_matrices.values():
    G = nx.from_numpy_array(matrix)
    G.remove_edges_from(nx.selfloop_edges(G))
    dict_eigcentrality = nx.eigenvector_centrality_numpy(G)  # Dictionary of eigencentrality of each node
    list_eigcentrality = [dict_eigcentrality[node] for node in range(G.number_of_nodes())]
    eigcentralities.append(list_eigcentrality)

# Convert list of lists into 2D array and reshape it into NUM_SAMPLES x NUM_ATOMS
eigcentralities = np.array(eigcentralities).reshape(NUM_SAMPLES, -1)

# Concatenate the reshaped Coulomb matrices, their eigenvalues, eigenvectors, and eigcentralities
X_reshaped_with_eigfeature_with_eigcentrality = np.concatenate((X_reshaped_with_eigfeature, eigcentralities), axis=1).astype(np.float64)
print("Dimensions of X_reshaped_with_eigfeature_with_eigcentrality: {}".format(X_reshaped_with_eigfeature_with_eigcentrality.shape))



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# CCSD(T)/ccpvdz atomization energy (Exact)
y_CCSD = np.array([val["CCSD(T)"]["ccpvdz"] for val in data["targets"].values()])
y_CCSD_scaling_factor = np.max(np.absolute(y_CCSD))
y_CCSD_scaled = y_CCSD / y_CCSD_scaling_factor

# Low fidelity - HF/sto3g atomization energy
y_HF = np.array([val["HF"]["sto3g"] for val in data["targets"].values()])
y_HF_scaling_factor = np.max(np.absolute(y_HF))
y_HF_scaled = y_HF / y_HF_scaling_factor

# Medium fidelity - MP2/631g atomization energy
y_MP2_631g = np.array([val["MP2"]["631g"] for val in data["targets"].values()])
y_MP2_631g_scaling_factor = np.max(np.absolute(y_MP2_631g))
y_MP2_631g_scaled = y_MP2_631g / y_MP2_631g_scaling_factor

# High fidelity - MP2/ccpvdz atomization energy
y_MP2_ccpvdz = np.array([val["MP2"]["ccpvdz"] for val in data["targets"].values()])
y_MP2_ccpvdz_scaling_factor = np.max(np.absolute(y_MP2_ccpvdz))
y_MP2_ccpvdz_scaled = y_MP2_ccpvdz / y_MP2_ccpvdz_scaling_factor

# Define your random seed
RANDOM_SEED = 42

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, indices_train, indices_test = train_test_split(X_reshaped_with_eigfeature_with_eigcentrality, range(len(X_reshaped_with_eigfeature_with_eigcentrality)), test_size=0.2, random_state=RANDOM_SEED)

# Use the same split indices to split all the y data
y_train_HF_scaled = y_HF_scaled[indices_train]
y_test_HF_scaled = y_HF_scaled[indices_test]

y_train_MP2_631g_scaled = y_MP2_631g_scaled[indices_train]
y_test_MP2_631g_scaled = y_MP2_631g_scaled[indices_test]

y_train_MP2_ccpvdz_scaled = y_MP2_ccpvdz_scaled[indices_train]
y_test_MP2_ccpvdz_scaled = y_MP2_ccpvdz_scaled[indices_test]

y_train_CCSD_scaled = y_CCSD_scaled[indices_train]
y_test_CCSD_scaled = y_CCSD_scaled[indices_test]
import time
# Define models for each fidelity level
start_time = time.time()
model_HF = GradientBoostingRegressor(random_state=RANDOM_SEED)
model_MP2_631g = GradientBoostingRegressor(random_state=RANDOM_SEED)
model_MP2_ccpvdz = GradientBoostingRegressor(random_state=RANDOM_SEED)

# Define models for discrepancies
model_HF_MP2_631g = GradientBoostingRegressor(random_state=RANDOM_SEED)
model_MP2_631g_MP2_ccpvdz = GradientBoostingRegressor(random_state=RANDOM_SEED)
model_MP2_ccpvdz_CCSD = GradientBoostingRegressor(random_state=RANDOM_SEED)

# Train the models on the lower fidelity data
model_HF.fit(X_train, y_train_HF_scaled)
model_MP2_631g.fit(X_train, y_train_MP2_631g_scaled)
model_MP2_ccpvdz.fit(X_train, y_train_MP2_ccpvdz_scaled)
end_time = time.time()
training_time = end_time - start_time
print(f"Training time for HF model: {training_time} seconds")
# Train the discrepancy models
model_HF_MP2_631g.fit(X_train, y_train_MP2_631g_scaled - y_train_HF_scaled)
model_MP2_631g_MP2_ccpvdz.fit(X_train, y_train_MP2_ccpvdz_scaled - y_train_MP2_631g_scaled)
model_MP2_ccpvdz_CCSD.fit(X_train, y_train_CCSD_scaled - y_train_MP2_ccpvdz_scaled)

# Function to predict atomization energy
def predict_atomization_energy(new_samples):
    return (model_HF.predict(new_samples) * y_HF_scaling_factor +
            model_HF_MP2_631g.predict(new_samples) * y_HF_scaling_factor +
            model_MP2_631g_MP2_ccpvdz.predict(new_samples) * y_MP2_631g_scaling_factor +
            model_MP2_ccpvdz_CCSD.predict(new_samples) * y_MP2_ccpvdz_scaling_factor)

# Predictions for test data
y_pred_multi = predict_atomization_energy(X_test)
y_pred_high = model_MP2_ccpvdz.predict(X_test) * y_MP2_ccpvdz_scaling_factor

# Metrics
mae_multi = mean_absolute_error(y_test_CCSD_scaled * y_CCSD_scaling_factor, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test_CCSD_scaled * y_CCSD_scaling_factor, y_pred_multi))
mae_high = mean_absolute_error(y_test_MP2_ccpvdz_scaled * y_MP2_ccpvdz_scaling_factor, y_pred_high)
rmse_high = np.sqrt(mean_squared_error(y_test_MP2_ccpvdz_scaled * y_MP2_ccpvdz_scaling_factor, y_pred_high))

print(f'Multi-fidelity model MAE: {mae_multi}, RMSE: {rmse_multi}')
print(f'High fidelity model MAE: {mae_high}, RMSE: {rmse_high}')
import pandas as pd
import time
# The fractions of high-fidelity data you want to use
fractions = [1]

results = []

for frac in fractions:
    # Select a fraction of the high-fidelity data
    num_samples = int(frac * len(y_train_MP2_ccpvdz_scaled))
    indices = np.random.choice(len(y_train_MP2_ccpvdz_scaled), size=num_samples, replace=False)

    X_train_high = X_train[indices]
    y_train_high = y_train_MP2_ccpvdz_scaled[indices]
    #Time the training session
    start_time = time.time()
    
    # Fine-tune the model on the selected high-fidelity data
    model_MP2_ccpvdz.fit(X_train_high, y_train_high)
    
    #Calculate how long training took
    training_time = time.time() - start_time
    # Predictions for test data
    y_pred_high = model_MP2_ccpvdz.predict(X_test) * y_MP2_ccpvdz_scaling_factor

    # Metrics
    mae_high = mean_absolute_error(y_test_MP2_ccpvdz_scaled * y_MP2_ccpvdz_scaling_factor, y_pred_high)
    rmse_high = np.sqrt(mean_squared_error(y_test_MP2_ccpvdz_scaled * y_MP2_ccpvdz_scaling_factor, y_pred_high))

    results.append((frac, mae_high, rmse_high, training_time))

# Convert results to a DataFrame for easy viewing
df_results = pd.DataFrame(results, columns=['Fraction', 'MAE', 'RMSE','Training time'])
print(df_results)

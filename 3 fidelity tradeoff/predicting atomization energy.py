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
y = np.array([val["MP2"]["ccpvdz"] for val in data["targets"].values()])
y_scaling_factor = np.max(np.absolute(y))
y_scaled = y/y_scaling_factor

# Constants
NUM_SAMPLES = len(sorted_matrices)
RANDOM_SEED = 42


# List of keys (molecule IDs)
ids = list(sorted_matrices.keys())

# Randomly select a single ID
selected_id = np.random.choice(ids)
print(f"Selected ID: {selected_id}")

# Get corresponding matrix
selected_matrix = sorted_matrices[selected_id]

# Initialize a plot
fig, ax = plt.subplots(figsize=(6, 6))

# Generate the graph
G = nx.from_numpy_array(selected_matrix)
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)
G.remove_edges_from(nx.selfloop_edges(G))

# Draw the network
nx.draw_networkx(G, nx.spring_layout(G), ax=ax, node_color='r')

# Set title as the molecule ID
ax.set_title("Molecule ID: H8C4N2O")

plt.show()

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

import xgboost as xgb
import time

# Set up XGBoost parameters and data
start_time = time.time()

NUM_FOLDS = 5
EARLY_STOPPING = 20
NUM_EPOCHS = 50

xg_data = xgb.DMatrix(X_reshaped_with_eigfeature_with_eigcentrality, label=y_scaled)

params = {
    "objective": "reg:squarederror",
    "seed": RANDOM_SEED,
    "eval_metric": "mae",
    "booster": "gbtree",
    "n_threads": -1
}

# Cross-validation with XGBoost
xgb_cv = xgb.cv(
    params,
    xg_data,
    num_boost_round=NUM_EPOCHS,
    nfold=NUM_FOLDS,
    early_stopping_rounds=EARLY_STOPPING,
    verbose_eval=0,
    seed=RANDOM_SEED,
    as_pandas=False,
    shuffle=True
)

print("Cross-validation done in %s seconds" % (time.time() - start_time))

# Performance evaluation
final_test_mae_mean = np.min(xgb_cv['test-mae-mean'])
print("Cross-validation loss: {}".format(final_test_mae_mean*y_scaling_factor))

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.plot([x*y_scaling_factor for x in xgb_cv['train-mae-mean']], label='Train loss')
plt.plot([x*y_scaling_factor for x in xgb_cv['test-mae-mean']], label='Test loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.show()



from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
import numpy as np
import time

# Define the metrics
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# Define cross-validation and data
RANDOM_SEED = 42
NUM_FOLDS = 5
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# XGBoost
start_time = time.time()
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', seed=RANDOM_SEED, eval_metric='mae', booster='gbtree', n_jobs=-1, n_estimators=100, learning_rate=0.1)
xgb_preds = cross_val_predict(xgb_model, X, y, cv=kf)
xgb_mae, xgb_rmse = compute_metrics(y, xgb_preds)
xgb_time = time.time() - start_time

# SVM
start_time = time.time()
svr_model = SVR(kernel='rbf', gamma=1e-4, epsilon=1e-6)
svr_preds = cross_val_predict(svr_model, X, y, cv=kf)
svr_mae, svr_rmse = compute_metrics(y, svr_preds)
svr_time = time.time() - start_time

# Gradient Boosting
start_time = time.time()
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_SEED, loss='squared_error')
gbr_preds = cross_val_predict(gbr_model, X, y, cv=kf)
gbr_mae, gbr_rmse = compute_metrics(y, gbr_preds)
gbr_time = time.time() - start_time

print("Model                 | MAE   | RMSE  | Time (s)")
print("-"*50)
print(f"XGBoost               | {xgb_mae:.3f} | {xgb_rmse:.3f} | {xgb_time:.3f}")
print(f"SVM                   | {svr_mae:.3f} | {svr_rmse:.3f} | {svr_time:.3f}")
print(f"Gradient Boosting     | {gbr_mae:.3f} | {gbr_rmse:.3f} | {gbr_time:.3f}")
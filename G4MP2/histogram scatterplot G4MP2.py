import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
with open('C:/Users/ragha/Downloads/molecule_data/G4MP2.json', 'r') as f:
    data = json.load(f)

g4mp2_data = pd.DataFrame(data['G4MP2'])
b3lyp_data = pd.DataFrame(data['B3LYP'])

# Plot histograms of the energies for both data sets
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
sns.histplot(g4mp2_data['U0'], ax=ax1, kde=True)
ax1.set_title('G4MP2 (High Fidelity)')
sns.histplot(b3lyp_data['U0'], ax=ax2, kde=True)
ax2.set_title('B3LYP (Low Fidelity)')
plt.show()

# Plot scatter plots of the energies for both data sets
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
sns.scatterplot(x='index', y='U0', data=g4mp2_data.reset_index(), ax=ax1)
ax1.set_title('G4MP2 (High Fidelity)')
sns.scatterplot(x='index', y='U0', data=b3lyp_data.reset_index(), ax=ax2)
ax2.set_title('B3LYP (Low Fidelity)')
plt.show()

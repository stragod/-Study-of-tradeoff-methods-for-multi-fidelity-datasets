# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:23:15 2023

@author: ragha
"""

import matplotlib.pyplot as plt

# Assuming you have a dictionary with your data
data = {
    "HF/sto-3g": {"Average Error": 15.655356, "Estimated Cost": 1.0},
    "MP2/6-31g": {"Average Error": 6.230508, "Estimated Cost": 56.942420},
    "CCSD(T)/cc-pvdz": {"Average Error": 0.0, "Estimated Cost": 5102.040816}
    
}

# Extract x (cost) and y (error) values
costs = [v["Estimated Cost"] for v in data.values()]
errors = [v["Average Error"] for v in data.values()]

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(costs, errors)

# Add labels for each point
for label, x, y in zip(data.keys(), costs, errors):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(-10,10), ha='center')

plt.xlabel('Estimated Cost (Normalized)')
plt.ylabel('Average Error')
plt.title('Error-Cost Analysis')
plt.grid(True)
plt.show()


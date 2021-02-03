"""
Model for solving a transportation problem:
minimize the total transportation cost for satisfying demand at
customers, from capacitated facilities.
Data:
    I - set of customers
    J - set of facilities
    unit_costs[i,j] - unit transportation cost on arc (i,j)
    demand[i] - demand at node i
    supply[j] - capacity
"""

from pyscipopt import Model, quicksum
import numpy as np

m = 100  # Number of customers
n = 100  # Number of suppliers

demand = np.random.randint(100, 500, m)

supply = np.random.randint(400, 500, n)
supply += (np.sum(demand) - np.sum(supply)) // n + 1  # Ensures just-barely feasiblity

assert np.sum(supply) >= np.sum(demand)

unit_costs = np.random.randint(1, 10, (m, n))

model = Model()

# Create variables
x = np.array([
    [model.addVar(vtype="i") for j in range(n)]
    for i in range(m)
])
# Demand constraints
for i in range(m):
    model.addCons(np.sum(x[i, :]) == demand[i])

# Capacity constraints
for j in range(n):
    model.addCons(np.sum(x[:, j]) <= supply[j])

# Objective
model.setObjective(np.sum(unit_costs * x))
model.optimize()
print("Optimal value:", model.getObjVal())

# Postprocess
x_opt = np.array([
    [model.getVal(x[i, j]) for j in range(n)]
    for i in range(m)
])

print(f"x_opt sparsity:\n{'_' * (n + 2)}")
for i in range(m):
    line = "".join(["*" if x_opt[i, j] >= 1e-6 else " " for j in range(n)])
    print("|" + line + "|")
print('-' * (n + 2))

from pyscipopt import Model
import numpy as np

n = 10000

values = np.random.uniform(1, 5, n)
weights = np.random.uniform(1, 5, n)

capacity = np.sum(weights) * 0.2

model = Model()

x = np.array([model.addVar(vtype="i") for i in range(n)])

model.setObjective(-np.sum(values * x))
model.addCons(np.sum(weights * x) <= capacity)
for i in range(n):
    model.addCons(0 <= (x[i] <= 1))

model.optimize()

# print("Items to take:")
# for i in range(n):
#     print(f"\tItem {i:3}: {'*' if model.getVal(x[i]) > 0.5 else ' '}")
import numpy as np

"""
Problem:

minimize x_1 + 5x_2 - 2x_3

subject to:
x_1 + x_2 + x_3 <= 4
x_1 <= 2
x_3 <= 3
3x_2 + x_3 <= 6
x_1, x_2, x_3 >= 0

"""

c = np.array([1, 5, -2, 0, 0, 0, 0])
A = np.array([
    [1, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 3, 1, 0, 0, 0, 1]
])
b = np.array([4, 2, 3, 6])

m, n = A.shape

### Initial Point
basic_indices = [0, 2, 5, 6]  # start with slack variables as basis
basic_indices = [3, 4, 5, 6]  # start with slack variables as basis
# basic_indices = [2, 4, 5, 6]

iteration = 0
while True:
    iteration += 1
    print(f"Iteration {iteration}:")
    B = A[:, basic_indices]
    x = np.zeros(n)
    x[basic_indices] = np.linalg.solve(B, b)
    print(f"\tx = {x}")
    if np.any(x < 0):
        raise RuntimeError("Initial point is not a BFS!")

    reduced_costs = c - c[basic_indices] @ np.linalg.inv(B) @ A
    print(f"\treduced_costs = {reduced_costs}")
    if np.all(reduced_costs >= 0):
        print("\tOptimality condition satisfied!")
        break
    entering_index = int(np.argwhere(reduced_costs < 0.)[0])  # Get the first index with negative reduced cost

    basic_direction = np.zeros(n)
    basic_direction[basic_indices] = -np.linalg.inv(B) @ A[:, entering_index].flatten()
    print(f"\tbasic_direction = {basic_direction}")
    if np.all(basic_direction >= 0):
        raise RuntimeError("\tProblem is unbounded!")
    basic_direction[entering_index] = 1

    with np.errstate(divide='ignore', invalid='ignore'):
        possible_step_sizes = np.where(
            basic_direction < 0,
            -x / basic_direction,
            np.Inf
        )
    exiting_index = np.argmin(possible_step_sizes)
    step_size = possible_step_sizes[exiting_index]
    print(f"\tstep_size = {step_size} (at index {exiting_index})")

    # Update point and basis
    x = x + step_size * basic_direction
    basic_indices = [
        index if index != exiting_index else entering_index
        for index in basic_indices
    ]

x_opt = x
f_opt = c @ x
print(f"x_opt = {x_opt}")
print(f"f_opt = {f_opt}")

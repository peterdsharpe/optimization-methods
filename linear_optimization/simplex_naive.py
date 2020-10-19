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

def simplex_naive(
        c=c,
        A=A,
        b=b,
        basic_indices=[3, 4, 5, 6],
        max_iter=100
):
    iteration = 0
    while True:  # Iterate
        ###
        iteration += 1
        if iteration > max_iter:
            raise OverflowError("Maximum number of iterations exceeded.")
        print(f"Iteration {iteration}:")

        ### Make the basis matrix
        print(f"\tbasic_indices = {basic_indices}")  # Display the basic indices
        B = A[:, basic_indices]

        ### Calculate the basic solution associated with the basis
        x = np.zeros(n)
        try:
            x[basic_indices] = np.linalg.solve(B, b)
        except np.linalg.LinAlgError:
            raise RuntimeError("Basis is not invertible!")
        print(f"\tx = {x}")  # Display the basic solution

        ### Check if the basic solution is feasible
        if np.any(x < 0):
            raise RuntimeError("Initial point is not a BFS!")

        ### Calculate the reduced costs
        reduced_costs = c - c[basic_indices] @ np.linalg.inv(B) @ A
        print(f"\treduced_costs = {reduced_costs}")

        ### Check if the solution is already optimal; if so, end here.
        if np.all(reduced_costs >= 0):
            print("\tOptimality condition satisfied!")
            break

        ### Choose which new index should enter the basis
        for index in range(n):  # Get the first non-basic index with negative reduced cost
            if reduced_costs[index] < 0 and index not in basic_indices:
                entering_index = index
                break

        ### Calculate the basic direction associated with the new entering index
        basic_direction = np.zeros(n)
        basic_direction[basic_indices] = -np.linalg.inv(B) @ A[:, entering_index].flatten()
        basic_direction[entering_index] = 1
        print(f"\tbasic_direction = {basic_direction}")

        ### Check if the basic direction indicates an unbounded problem
        if np.all(basic_direction >= 0):
            raise RuntimeError("\tProblem is unbounded!")

        ### Calculate the step size associated with each possible choice of exiting index
        with np.errstate(divide='ignore', invalid='ignore'):
            possible_step_sizes = np.where(
                basic_direction < 0,
                -x / basic_direction,
                np.Inf
            )

        ### Choose the index that should exit by looking at which index breaks feasibility first
        exiting_index = np.argmin(possible_step_sizes)

        ### Calculate the associated step size with this exiting index
        step_size = possible_step_sizes[exiting_index]
        print(f"\tstep_size = {step_size} (at index {exiting_index})")

        ### Update point and basis
        x = x + step_size * basic_direction
        basic_indices = [
            index if index != exiting_index else entering_index
            for index in basic_indices
        ]

    ### Calculate the value of the optimal solution
    x_opt = x
    f_opt = c @ x
    print(f"x_opt = {x_opt}")
    print(f"f_opt = {f_opt}")

    return (x_opt, f_opt)


if __name__ == '__main__':
    simplex_naive()

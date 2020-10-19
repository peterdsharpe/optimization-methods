import pytest
from simplex_naive import *
from itertools import combinations

c = np.array([1, 5, -2, 0, 0, 0, 0])
A = np.array([
    [1, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 3, 1, 0, 0, 0, 1]
])
b = np.array([4, 2, 3, 6])

m, n = A.shape

def test_simplex_naive():
    basic_indices_combinations = combinations(range(n), 4)
    for basic_indices in basic_indices_combinations:
        basic_indices = list(basic_indices)
        print("-"*10)
        print(f"basic_indices = {basic_indices}")
        try:
            x_opt, f_opt = simplex_naive(basic_indices=basic_indices)
            assert f_opt == pytest.approx(-6)
        except RuntimeError as e:
            print(e)


if __name__ == '__main__':
    test_simplex_naive()

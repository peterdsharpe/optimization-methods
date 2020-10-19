import pytest
from simplex_naive import *
from itertools import combinations


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

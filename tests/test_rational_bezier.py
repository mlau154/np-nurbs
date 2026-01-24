"""
Tests Bézier curve and surface evaluation functions
against the ``rust_nurbs`` library for correctness
"""
import pytest

import numpy as np
import np_nurbs
import rust_nurbs


@pytest.fixture
def p_curve() -> np.ndarray[tuple[int, int], np.dtype[float]]:
    return np.random.uniform(low=-5.0, high=5.0, size=(10, 3))


@pytest.fixture
def p_surf() -> np.ndarray[tuple[int, int, int], np.dtype[float]]:
    return np.random.uniform(low=-5.0, high=5.0, size=(10, 10, 3))


@pytest.fixture
def w_curve() -> np.ndarray[tuple[int,], np.dtype[float]]:
    return np.random.uniform(low=0.01, high=10.0, size=(10,))


@pytest.fixture
def w_surf() -> np.ndarray[tuple[int, int], np.dtype[float]]:
    return np.random.uniform(low=0.01, high=10.0, size=(10, 10))


def test_rational_bezier_curve_eval_grid(
        p_curve: np.ndarray[tuple[int, int], np.dtype[float]],
        w_curve: np.ndarray[tuple[int,], np.dtype[float]]
        ):
    np_curve = np_nurbs.rational_bezier_curve_eval_grid(
            p_curve, w_curve, 150
            )
    rust_curve = np.array(rust_nurbs.rational_bezier_curve_eval_grid(
            p_curve, w_curve, 150
            ))
    assert np.all(np.isclose(np_curve, rust_curve))


def test_rational_bezier_surf_eval_grid(
        p_surf: np.ndarray[tuple[int, int, int], np.dtype[float]],
        w_surf: np.ndarray[tuple[int, int], np.dtype[float]]
        ):
    np_surf = np_nurbs.rational_bezier_surf_eval_grid(
            p_surf, w_surf, 50, 50
            )
    rust_surf = np.array(rust_nurbs.rational_bezier_surf_eval_grid(
            p_surf, w_surf, 50, 50
            ))
    assert np.all(np.isclose(np_surf, rust_surf))


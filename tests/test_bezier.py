"""
Tests Bézier curve and surface evaluation functions
against the ``rust_nurbs`` library for correctness
"""
import pytest

import numpy as np
import np_nurbs
import rust_nurbs


@pytest.fixture
def p_curve() -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    return np.random.uniform(low=-5.0, high=5.0, size=(10, 3))


@pytest.fixture
def p_surf() -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
    return np.random.uniform(low=-5.0, high=5.0, size=(10, 10, 3))


def test_bezier_curve_eval_grid(
        p_curve: np.ndarray[tuple[int, int], np.dtype[np.float64]]
        ):
    np_curve = np_nurbs.bezier_curve_eval_grid(
            p_curve, 150
            )
    rust_curve = np.array(rust_nurbs.bezier_curve_eval_grid(
            p_curve, 150
            ))
    assert np.all(np.isclose(np_curve, rust_curve))


def test_bezier_curve_dcdt_grid(
        p_curve: np.ndarray[tuple[int, int], np.dtype[np.float64]]
        ):
    np_curve = np_nurbs.bezier_curve_dcdt_grid(
            p_curve, 150
            )
    rust_curve = np.array(rust_nurbs.bezier_curve_dcdt_grid(
            p_curve, 150
            ))
    assert np.all(np.isclose(np_curve, rust_curve))


def test_bezier_curve_d2cdt2_grid(
        p_curve: np.ndarray[tuple[int, int], np.dtype[np.float64]]
        ):
    np_curve = np_nurbs.bezier_curve_d2cdt2_grid(
            p_curve, 150
            )
    rust_curve = np.array(rust_nurbs.bezier_curve_d2cdt2_grid(
            p_curve, 150
            ))
    assert np.all(np.isclose(np_curve, rust_curve))


def test_bezier_surf_eval_grid(
        p_surf: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
        ):
    np_surf = np_nurbs.bezier_surf_eval_grid(
            p_surf, 50, 50
            )
    rust_surf = np.array(rust_nurbs.bezier_surf_eval_grid(
            p_surf, 50, 50
            ))
    assert np.all(np.isclose(np_surf, rust_surf))


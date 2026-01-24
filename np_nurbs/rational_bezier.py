import numpy as np

from np_nurbs import coefficient_matrices


__all__ = [
    "rational_bezier_curve_eval_grid",
    "rational_bezier_surf_eval_grid"
]


def rational_bezier_curve_eval_grid(
        p: np.ndarray[tuple[int, int], np.dtype[float]], 
        w: np.ndarray[tuple[int,], np.dtype[float]],
        nt: int,
        ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
    """
    Evaluates a rational Bézier curve on an evenly spaced parameter vector
    (``linspace(0, 1, nt)``) using a fully vectorized formulation.

    Parameters
    ----------
    p: np.ndarray[tuple[int, int], np.dtype[float]]
        Rational Bézier curve control point array. 
        This array has shape
        :math:`(n+1) \\times d`, where :math:`n` is
        the curve degree and :math:`d` is the number
        of dimensions (usually 2 or 3)
    w: np.ndarray[tuple[int,], np.dtype[float]]
        Vector of weights, corresponding one-to-one with
        the control points
    nt: int
        Number of evenly spaced parameters at which to
        evaluate the curve

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[float]]
        The evaluated rational Bézier curve with shape
        :math:`n_t \\times d`, where :math:`n_t`
        is the number of parameters
    """
    assert len(p) == len(w)
    degree = len(p) - 1
    t = np.linspace(0.0, 1.0, nt, dtype=float)
    powers = (degree - np.arange(degree + 1))[:, np.newaxis]
    t_mat = t ** powers
    
    m = coefficient_matrices[degree]

    # Homogeneous control points
    pw = np.insert(p, p.shape[-1], 1.0, axis=1)
    pw = pw * w[:, np.newaxis]

    a = np.dot(t_mat.T, m)
    b = np.dot(a, pw)

    return b[:, :-1] / b[:, -1][:, np.newaxis]


def rational_bezier_surf_eval_grid(
        p: np.ndarray[tuple[int, int, int], np.dtype[float]],
        w: np.ndarray[tuple[int, int], np.dtype[float]],
        nu: int, 
        nv: int,
        ) -> np.ndarray[tuple[int, int, int], np.dtype[float]]:
    """
    Evaluates a rational Bézier surface on a uniform parameter grid
    (``linspace(0, 1, nu), linspace(0, 1, nv)``) 
    using a fully vectorized formulation.

    Parameters
    ----------
    p: np.ndarray[tuple[int, int, int], np.dtype[float]]
        Rational Bézier surface control point array. 
        This array has shape
        :math:`(n+1) \\times (m+1) \\times d`, where :math:`n` is
        the surface degree in the :math:`u`-direction,
        :math:`m` is the surface degree in the :math:`v`-direction,
        and :math:`d` is the number of dimensions (usually 3)
    w: np.ndarray[tuple[int, int], np.dtype[float]]
        Vector of weights, corresponding one-to-one with
        the control points
    nu: int
        Number of evenly spaced parameters at which to
        evaluate the surface in the :math:`u`-direction
    nv: int
        Number of evenly spaced parameters at which to
        evaluate the surface in the :math:`v`-direction

    Returns
    -------
    np.ndarray[tuple[int, int, int], np.dtype[float]]
        The evaluated rational Bézier surface with shape
        :math:`n_u \\times n_v \\times d`, where :math:`n_u`
        is the number of parameters in the
        :math:`u`-direction and :math:`n_v` is the number
        of parameters in the :math:`v`-direction
    """
    n = p.shape[0] - 1
    m = p.shape[1] - 1
    u = np.linspace(0.0, 1.0, nu, dtype=float)
    v = np.linspace(0.0, 1.0, nv, dtype=float)
    u_powers = (n - np.arange(n + 1))[:, np.newaxis]
    v_powers = (m - np.arange(m + 1))[:, np.newaxis]
    u_mat = u ** u_powers
    v_mat = v ** v_powers

    # Homogeneous control points
    pw = np.insert(p, p.shape[-1], 1.0, axis=2)
    pw = pw * w[:, :, np.newaxis]

    bu = np.dot(u_mat.T, coefficient_matrices[n])
    bv = np.dot(v_mat.T, coefficient_matrices[m])
    a = np.dot(bv, pw)
    b = np.dot(bu, a)
    return b[:, :, :-1] / b[:, :, -1][:, :, np.newaxis]


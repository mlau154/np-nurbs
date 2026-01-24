import numpy as np

from np_nurbs import coefficient_matrices


__all__ = [
    "bezier_curve_eval_grid",
    "bezier_curve_dcdt_grid",
    "bezier_surf_eval_grid",
]


def bezier_curve_eval_grid(
        p: np.ndarray[tuple[int, int], np.dtype[float]], 
        nt: int,
        ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
    """
    Evaluates a Bézier curve on an evenly spaced parameter vector
    (``linspace(0, 1, nt)``) using a fully vectorized formulation.

    Parameters
    ----------
    p: np.ndarray[tuple[int, int], np.dtype[float]]
        Bézier control point array. This array has shape
        :math:`(n+1) \\times d`, where :math:`n` is
        the curve degree and :math:`d` is the number
        of dimensions (usually 2 or 3)
    nt: int
        Number of evenly spaced parameters at which to
        evaluate the curve

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[float]]
        The evaluated Bézier curve with shape
        :math:`n_t \\times d`, where :math:`n_t`
        is the number of parameters
    """
    degree = len(p) - 1
    t = np.linspace(0.0, 1.0, nt, dtype=float)
    powers = (degree - np.arange(degree + 1))[:, np.newaxis]
    t_mat = t ** powers
    
    m = coefficient_matrices[degree]
    
    a = np.dot(t_mat.T, m)
    b = np.dot(a, p)
    
    return b


def bezier_curve_dcdt_grid(
        p: np.ndarray[tuple[int, int], np.dtype[float]],
        nt: int
        ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
    """
    Evaluates the first derivative of a Bézier curve with
    respect to its parameter :math:`t` on an evenly
    spaced parameter vector (``linspace(0, 1, nt)``)
    using a fully vectorized formulation.

    Parameters
    ----------
    p: np.ndarray[tuple[int, int], np.dtype[float]]
        Bézier control point array. This array has shape
        :math:`(n+1) \\times d`, where :math:`n` is
        the curve degree and :math:`d` is the number
        of dimensions (usually 2 or 3)
    nt: int
        Number of evenly spaced parameters at which to
        evaluate the first derivative

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[float]]
        The evaluated Bézier curve first derivative with shape
        :math:`n_t \\times d`, where :math:`n_t`
        is the number of parameters
    """
    degree = len(p) - 1
    t = np.linspace(0.0, 1.0, nt, dtype=float)
    powers = (degree - 1 - np.arange(degree + 1 - 1))[:, np.newaxis]
    t_mat = t ** powers
    
    m = coefficient_matrices[degree - 1]

    p_diff = np.diff(p, axis=0)
    
    a = np.dot(t_mat.T, m)
    b = degree * np.dot(a, p_diff)
    
    return b


def bezier_surf_eval_grid(
        p: np.ndarray[tuple[int, int, int], np.dtype[float]],
        nu: int, 
        nv: int,
        ) -> np.ndarray[tuple[int, int, int], np.dtype[float]]:
    """
    Evaluates a Bézier surface on a uniform parameter grid
    (``linspace(0, 1, nu), linspace(0, 1, nv)``) 
    using a fully vectorized formulation.

    Parameters
    ----------
    p: np.ndarray[tuple[int, int, int], np.dtype[float]]
        Bézier surface control point array. This array has shape
        :math:`(n+1) \\times (m+1) \\times d`, where :math:`n` is
        the surface degree in the :math:`u`-direction,
        :math:`m` is the surface degree in the :math:`v`-direction,
        and :math:`d` is the number of dimensions (usually 3)
    nu: int
        Number of evenly spaced parameters at which to
        evaluate the surface in the :math:`u`-direction
    nv: int
        Number of evenly spaced parameters at which to
        evaluate the surface in the :math:`v`-direction

    Returns
    -------
    np.ndarray[tuple[int, int, int], np.dtype[float]]
        The evaluated Bézier surface with shape
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
    bu = np.dot(u_mat.T, coefficient_matrices[n])
    bv = np.dot(v_mat.T, coefficient_matrices[m])
    a = np.dot(bv, p)
    b = np.dot(bu, a)
    return b


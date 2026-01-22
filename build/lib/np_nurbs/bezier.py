import numpy as np


def bezier_curve_eval_grid(p: np.ndarray, nt: int):
    degree = len(p) - 1
    t = np.linspace(0.0, 1.0, nt, dtype=float)
    powers = (degree - np.arange(degree + 1))[:, np.newaxis]
    t_mat = t ** powers
    
    m = coefficient_matrices[degree]
    
    a = np.dot(t_mat.T, m)
    b = np.dot(a, p)
    
    return b


def bezier_surf_eval_grid(p: np.ndarray[tuple[int, int, int], np.dtype[float], nu: int, nv: int):
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


from statistics import mean, stdev
import time

import numpy as np
import rust_nurbs

from np_nurbs import *


def compare_bezier_curve_eval_timings():
    p_rand = np.random.uniform(low=0.0, high=1.0, size=(1000, 5, 3))

    cupy_timings, numpy_timings, rnurbs_timings = [], [], []
    m_gpu, m = None, None
    for p in p_rand:
        start_time = time.perf_counter()
        bezier_curve_eval_grid(p, 150)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        numpy_timings.append(elapsed_time)
    for p in p_rand:
        start_time = time.perf_counter()
        rust_nurbs.bezier_curve_eval_grid(p, 150)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        rnurbs_timings.append(elapsed_time)

    mean_time_numpy = mean(numpy_timings[1:])
    std_dev_time_numpy = stdev(numpy_timings[1:])
    mean_time_rnurbs = mean(rnurbs_timings[1:])
    std_dev_time_rnurbs = stdev(rnurbs_timings[1:])

    print(f"Bézier curve evaluation...")
    print(f"Mean execution time: {mean_time_numpy:.6f} s (np_nurbs) | {mean_time_rnurbs:.6f} s (rust_nurbs) | Ratio -- {mean_time_rnurbs / mean_time_numpy:.1f}")
    print(f"Standard deviation: {std_dev_time_numpy:.6f} s (np_nurbs) | {std_dev_time_rnurbs:.6f} s (rust_nurbs)")


def compare_bezier_curve_dcdt_grid_timings():
    p_rand = np.random.uniform(low=0.0, high=1.0, size=(1000, 5, 3))

    cupy_timings, numpy_timings, rnurbs_timings = [], [], []
    m_gpu, m = None, None
    for p in p_rand:
        start_time = time.perf_counter()
        bezier_curve_dcdt_grid(p, 150)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        numpy_timings.append(elapsed_time)
    for p in p_rand:
        start_time = time.perf_counter()
        rust_nurbs.bezier_curve_dcdt_grid(p, 150)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        rnurbs_timings.append(elapsed_time)

    mean_time_numpy = mean(numpy_timings[1:])
    std_dev_time_numpy = stdev(numpy_timings[1:])
    mean_time_rnurbs = mean(rnurbs_timings[1:])
    std_dev_time_rnurbs = stdev(rnurbs_timings[1:])

    print(f"Bézier curve first derivative evaluation...")
    print(f"Mean execution time: {mean_time_numpy:.6f} s (np_nurbs) | {mean_time_rnurbs:.6f} s (rust_nurbs) | Ratio -- {mean_time_rnurbs / mean_time_numpy:.1f}")
    print(f"Standard deviation: {std_dev_time_numpy:.6f} s (np_nurbs) | {std_dev_time_rnurbs:.6f} s (rust_nurbs)")


def compare_bezier_surf_eval_timings():
    p_rand = np.random.uniform(low=-3.0, high=3.0, size=(100, 5, 6, 3))
    numpy_timings, rnurbs_timings = [], []
    for p in p_rand:
        start_time = time.perf_counter()
        bezier_surf_eval_grid(p, 75, 75)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        numpy_timings.append(elapsed_time)
    for p in p_rand:
        start_time = time.perf_counter()
        rust_nurbs.bezier_surf_eval_grid(p, 75, 75)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        rnurbs_timings.append(elapsed_time)

    mean_time_numpy = mean(numpy_timings[1:])
    std_dev_time_numpy = stdev(numpy_timings[1:])
    mean_time_rnurbs = mean(rnurbs_timings[1:])
    std_dev_time_rnurbs = stdev(rnurbs_timings[1:])

    print(f"Bézier surface evaluation...")
    print(f"Mean execution time: {mean_time_numpy:.6f} s (np_nurbs) | {mean_time_rnurbs:.6f} s (rust_nurbs) | Ratio -- {mean_time_rnurbs / mean_time_numpy:.1f}")
    print(f"Standard deviation: {std_dev_time_numpy:.6f} s (np_nurbs) | {std_dev_time_rnurbs:.6f} s (rust_nurbs)")


def compare_rational_bezier_curve_eval_timings():
    p_rand = np.random.uniform(low=0.0, high=1.0, size=(1000, 5, 3))
    w_rand = np.random.uniform(low=0.01, high=10.0, size=(1000, 5))

    cupy_timings, numpy_timings, rnurbs_timings = [], [], []
    m_gpu, m = None, None
    for p, w in zip(p_rand, w_rand):
        start_time = time.perf_counter()
        rational_bezier_curve_eval_grid(p, w, 150)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        numpy_timings.append(elapsed_time)
    for p, w in zip(p_rand, w_rand):
        start_time = time.perf_counter()
        rust_nurbs.rational_bezier_curve_eval_grid(p, w, 150)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        rnurbs_timings.append(elapsed_time)

    mean_time_numpy = mean(numpy_timings[1:])
    std_dev_time_numpy = stdev(numpy_timings[1:])
    mean_time_rnurbs = mean(rnurbs_timings[1:])
    std_dev_time_rnurbs = stdev(rnurbs_timings[1:])

    print(f"Rational Bézier curve evaluation...")
    print(f"Mean execution time: {mean_time_numpy:.6f} s (np_nurbs) | {mean_time_rnurbs:.6f} s (rust_nurbs) | Ratio -- {mean_time_rnurbs / mean_time_numpy:.1f}")
    print(f"Standard deviation: {std_dev_time_numpy:.6f} s (np_nurbs) | {std_dev_time_rnurbs:.6f} s (rust_nurbs)")


def compare_rational_bezier_surf_eval_timings():
    p_rand = np.random.uniform(low=-3.0, high=3.0, size=(100, 5, 6, 3))
    w_rand = np.random.uniform(low=0.01, high=10.0, size=(100, 5, 6))
    numpy_timings, rnurbs_timings = [], []
    for p, w in zip(p_rand, w_rand):
        start_time = time.perf_counter()
        rational_bezier_surf_eval_grid(p, w, 75, 75)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        numpy_timings.append(elapsed_time)
    for p, w in zip(p_rand, w_rand):
        start_time = time.perf_counter()
        rust_nurbs.rational_bezier_surf_eval_grid(p, w, 75, 75)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        rnurbs_timings.append(elapsed_time)

    mean_time_numpy = mean(numpy_timings[1:])
    std_dev_time_numpy = stdev(numpy_timings[1:])
    mean_time_rnurbs = mean(rnurbs_timings[1:])
    std_dev_time_rnurbs = stdev(rnurbs_timings[1:])

    print(f"Rational Bézier surface evaluation...")
    print(f"Mean execution time: {mean_time_numpy:.6f} s (np_nurbs) | {mean_time_rnurbs:.6f} s (rust_nurbs) | Ratio -- {mean_time_rnurbs / mean_time_numpy:.1f}")
    print(f"Standard deviation: {std_dev_time_numpy:.6f} s (np_nurbs) | {std_dev_time_rnurbs:.6f} s (rust_nurbs)")


def main():
    compare_bezier_curve_eval_timings()
    compare_bezier_curve_dcdt_grid_timings()
    compare_bezier_surf_eval_timings()
    compare_rational_bezier_curve_eval_timings()
    compare_rational_bezier_surf_eval_timings()


if __name__ == "__main__":
    main()


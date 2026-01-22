import numpy as np
import rust_nurbs
from statistics import mean, stdev
import time

from np_nurbs.bezier import *


def main():
    # Curve accuracy test
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, -0.4],
        [1.0, 0.0]
    ])
    curves_close = bool(np.all(np.isclose(np.array(
        rust_nurbs.bezier_curve_eval_grid(p, 150)), bezier_curve_eval_grid(p, 150)
    )))
    print(f"{curves_close = }")
    
    # Surface accuracy test
    p = np.array([
        [[0.0, 0.0, 0.0],
         [0.3, 0.5, 0.1],
         [-0.1, 0.2, 0.8],
         [0.3, 0.8, -0.5]],
        [[0.8, 0.3, -0.3],
         [0.6, 0.4, 0.2],
         [0.1, 0.2, 0.3],
         [2.0, 3.0, 4.0]],
        [[0.3, 0.3, 0.3],
         [0.5, 0.4, 3.3],
         [1.0, 2.0, -0.5],
         [0.6, 0.8, 1.0]]
    ])
    surfs_close = bool(np.all(np.isclose(np.array(
        rust_nurbs.bezier_surf_eval_grid(p, 50, 30)), bezier_surf_eval_grid(p, 50, 30)
    )))
    print(f"{surfs_close = }")

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

    print(f"Curve evaluation...")
    print(f"Mean execution time (numpy): {mean_time_numpy:.6f} seconds")
    print(f"Standard deviation (numpy): {std_dev_time_numpy:.6f} seconds")
    print(f"Mean execution time (rnurbs): {mean_time_rnurbs:.6f} seconds")
    print(f"Standard deviation time (rnurbs): {std_dev_time_rnurbs:.6f} seconds")

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

    print(f"Surface evaluation...")
    print(f"Mean execution time (numpy): {mean_time_numpy:.6f} seconds")
    print(f"Standard deviation (numpy): {std_dev_time_numpy:.6f} seconds")
    print(f"Mean execution time (rnurbs): {mean_time_rnurbs:.6f} seconds")
    print(f"Standard deviation time (rnurbs): {std_dev_time_rnurbs:.6f} seconds")


if __name__ == "__main__":
    main()


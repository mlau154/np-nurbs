import time
from statistics import mean, stdev
import cupy as cp
import numpy as np
import rust_nurbs

m_gpu = cp.array([
    [1, 0, 0, 0],
    [-3, 3, 0, 0],
    [3, -6, 3, 0],
    [-1, 3, -3, 1]
], dtype=cp.float32)
m = np.array([
    [1, 0, 0, 0],
    [-3, 3, 0, 0],
    [3, -6, 3, 0],
    [-1, 3, -3, 1]
], dtype=np.float32)


def generate_gpu_coefficient_matrix(degree: int):
    matrix_size = degree + 1
    m = cp.zeros(shape=(matrix_size, matrix_size))
    for k in range(matrix_size):
        for i in range(matrix_size):
            if degree - k - i < 0:
                continue
            


def evaluate_bezier_grid(p: np.ndarray, nt: int):
    # Create time steps directly on the GPU
    t = cp.linspace(0.0, 1.0, nt, dtype=cp.float32)
    
    # Construct the matrix directly on GPU
    t_mat = cp.array([
        cp.ones(len(t)),
        t,
        t**2,
        t**3
    ], dtype=cp.float32)
    
    
    # Move points to GPU
    p_gpu = cp.array(p, dtype=cp.float32)
    
    # Use standard dot product (uses cuBLAS internally)
    a = cp.dot(t_mat.T, m_gpu) # Note .T for proper matrix alignment
    b = cp.dot(a, p_gpu)
    
    # Move back to CPU only when needed for printing
    # print(f"Result: {cp.asnumpy(b)}")


def evaluate_bezier_grid_numpy(p: np.ndarray, nt: int):
    # Create time steps directly on the GPU
    t = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    
    # Construct the matrix directly on GPU
    t_mat = np.array([
        np.ones(len(t)),
        t,
        t**2,
        t**3
    ], dtype=np.float32)
    
    
    # Use standard dot product (uses cuBLAS internally)
    a = np.dot(t_mat.T, m) # Note .T for proper matrix alignment
    b = np.dot(a, p)
    
    # Move back to CPU only when needed for printing
    # print(f"Result: {b}")


def main():
    # p = np.array([
    #     [0.0, 0.0, 0.0],
    #     [0.3, 0.5, 0.0],
    #     [0.7, -0.5, 0.0],
    #     [1.0, 0.0, 0.0]
    # ])
    p_rand = np.random.uniform(low=0.0, high=1.0, size=(1000, 4, 3))
    cupy_timings, numpy_timings, rnurbs_timings = [], [], []
    for p in p_rand:
        start_time = time.perf_counter()
        evaluate_bezier_grid(p, 150)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cupy_timings.append(elapsed_time)
    for p in p_rand:
        start_time = time.perf_counter()
        evaluate_bezier_grid_numpy(p, 150)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        numpy_timings.append(elapsed_time)
    for p in p_rand:
        start_time = time.perf_counter()
        rust_nurbs.bezier_curve_eval_grid(p, 150)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        rnurbs_timings.append(elapsed_time)

    mean_time_cupy = mean(cupy_timings[1:])
    std_dev_time_cupy = stdev(cupy_timings[1:]) # Uses the sample standard deviation (n-1 degrees of freedom)
    mean_time_numpy = mean(numpy_timings[1:])
    std_dev_time_numpy = stdev(numpy_timings[1:])
    mean_time_rnurbs = mean(rnurbs_timings[1:])
    std_dev_time_rnurbs = stdev(rnurbs_timings[1:])

    print(f"Mean execution time (cupy): {mean_time_cupy:.6f} seconds")
    print(f"Standard deviation (cupy): {std_dev_time_cupy:.6f} seconds")
    print(f"Mean execution time (numpy): {mean_time_numpy:.6f} seconds")
    print(f"Standard deviation (numpy): {std_dev_time_numpy:.6f} seconds")
    print(f"Mean execution time (rnurbs): {mean_time_rnurbs:.6f} seconds")
    print(f"Standard deviation time (rnurbs): {std_dev_time_rnurbs:.6f} seconds")


if __name__ == "__main__":
    main()



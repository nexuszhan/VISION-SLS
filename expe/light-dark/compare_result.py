import os
import numpy as np


def _is_close(a, b, rtol=1e-5, atol=1e-3):
    return a.shape == b.shape and np.allclose(a, b, atol=atol)


def main():
    base_path = os.path.dirname(__file__)
    file_a = os.path.join(base_path, "data/light-dark_SDP.npz")
    file_b = os.path.join(base_path, "data/light-dark.npz")

    data_a = np.load(file_a, mmap_mode="r")
    data_b = np.load(file_b, mmap_mode="r")

    keys = ("nominal_traj", "nominal_input", "backoff")
    results = {k: _is_close(data_a[k], data_b[k], atol=2e-3) for k in keys}

    for k in keys:
        print(f"{k}: {'close' if results[k] else 'not close'}")
    print(f"all_close: {all(results.values())}")

if __name__ == "__main__":
    main()

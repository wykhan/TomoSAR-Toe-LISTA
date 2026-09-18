"""Small end-to-end check for the data, solver, and CUDA training APIs."""

import os
import tempfile

import cvxpy as cp
import numpy as np
import scipy
import torch


def main():
    # paras12 writes D.npy during import; keep the smoke run out of the project tree.
    with tempfile.TemporaryDirectory() as workdir:
        os.chdir(workdir)
        from paras12 import D, M, N
        from algorithm_toe import Toe_LISTA_Ada, minL1_CVX, sparse_dataset_double

        torch.manual_seed(0)
        np.random.seed(0)
        dataset = sparse_dataset_double(N, 2, 4, 1, snr=5, A=D)
        assert dataset.X.shape == (4, N) and dataset.Y.shape == (4, M)

        gpu_index = 1 if torch.cuda.device_count() > 1 else 0
        device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
        model = Toe_LISTA_Ada(M, N, 2, np.array([1, 1])).to(device)
        x, y = dataset[:]
        target = x.float().to(device)
        yr, yi = y.real.float().to(device), y.imag.float().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        xr, xi = model(yr, yi)
        loss = torch.nn.functional.mse_loss(xr, target) + xi.square().mean()
        assert torch.isfinite(loss).item()
        loss.backward()
        assert model.Wre.grad is not None and torch.isfinite(model.Wre.grad).all().item()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Exercise the project's constrained L1 CVXPY path on a tiny exact problem.
        result = minL1_CVX(np.array([1.0, 0.0]), np.eye(2), 0.01)
        assert result.shape == (2,) and np.all(np.isfinite(result))
        print(f"PASS Python/torch {torch.__version__}, CUDA build {torch.version.cuda}, "
              f"device {device}, NumPy {np.__version__}, SciPy {scipy.__version__}, "
              f"CVXPY {cp.__version__}, loss {loss.item():.6f}")


if __name__ == "__main__":
    main()

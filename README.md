# TomoSAR-LISTA

## Runtime environment

The checked runtime uses Python 3.10, PyTorch 2.2.2+cu121, NumPy 1.26.4,
SciPy 1.13.0, and CVXPY 1.5.2. It was selected for an NVIDIA 550 driver and
does not require the local CUDA compiler (`nvcc`) for this project's Python code.
The remaining direct Python dependencies are pinned in `requirements.txt`.

```bash
conda create -n tomosar_lista python=3.10 -y
conda activate tomosar_lista
python -m pip install --index-url https://download.pytorch.org/whl/cu121 'torch==2.2.2+cu121'
python -m pip install -r requirements.txt
python -m pip check
python smoke_test.py
```

Run the training scripts from the repository directory. `paras12.py` writes
`res/data_8td_randA_randpphi_snr_train_k1_2/D.npy` on import; the data
generation script creates the larger training and testing `.pt` files needed
by `train_spare_alista.py`. The smoke test uses a temporary directory and
does not run the full data generation or training workload.

## Citation

If you find this code useful for your research, please cite our paper:

@article{ma2026toeplitz,
  title={Toeplitz-Structured Deep Unfolding Network for TomoSAR 3-D Reconstruction},
  author={Ma, Qian and Qian, Kun and Shen, Peng and et al.},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2026}
}


## Contact
If you have any questions, suggestions, or collaborations, please feel free to contact:
**Qian Ma**  
📧 Email: 2233809618@qq.com

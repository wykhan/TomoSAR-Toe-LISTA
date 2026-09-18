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

## Formal script smoke run

The data and training scripts accept `--data-dir` so their generated files and
`D.npy` stay together. The sample counts are per SNR or interval setting; the
original defaults remain 1280 training and 256 testing samples per setting.
Training defaults remain 10 epochs, batch size 128, validation every batch,
and a checkpoint every 10 epochs. `--device auto` selects the first available
CUDA device or CPU. An explicit device such as `cuda:1` is also supported.

```bash
python train_dataset_SNR20250318_k1-2.py --data-dir res/official_smoke --training-points 2 --testing-points 1
python train_spare_alista.py --data-dir res/official_smoke --epochs 1 --batch-size 16 --test-freq 1 --checkpoint-every 1 --device cuda:0
```

Verified on 2026-09-18 with the checked Python 3.10/PyTorch 2.2.2+cu121
environment and an RTX 4090. The data script saved 77 training and 39 testing
examples as `.pt` files. The training script processed four batches in one
epoch, reported a first-step `Wre` gradient norm of 0.0442709 and weight
change of 0.0155857, and completed four validation passes. Final batch
training loss was 0.0299405 and validation loss was 0.0308452. The checkpoint
`res/official_smoke/Toe_LISTA_Ada_epoch_0_8td_250319.pt` and loss CSV were
saved; the checkpoint was loaded again and its tensors were finite. The
generated `res/` directory is ignored by Git.

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

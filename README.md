## Physics-Informed Neural Networks with Extended Fiducial Inference (PyTorch)

This repository accompanies the NeurIPS 2025 paper “Uncertainty Quantification for Physics-Informed Neural Networks with Extended Fiducial Inference” and provides reference implementations for EFI-PINN, classic PINN (with dropout), and Bayesian PINN baselines in PyTorch.

- **Paper**: Uncertainty Quantification for Physics-Informed Neural Networks with Extended Fiducial Inference  
  Authors: Frank Shih, Zhenghao Jiang, Faming Liang  
  arXiv: [`2505.19136`](https://arxiv.org/abs/2505.19136), DOI: `10.48550/arXiv.2505.19136`  
  OpenReview: [`HFcQGutJJn`](https://openreview.net/forum?id=HFcQGutJJn)

---

## Installation

### Requirements
- torch
- hamiltorch
- scikit-learn
- numpy
- matplotlib
- seaborn
- ruamel.yaml
- tqdm

Install the package (editable) and Python dependencies:

```bash
pip install --use-pep517 -e .
pip install -r requirements.txt
```

---

## Quick Start

### EFI-PINN (proposed)
```bash
python train.py --algo pinn_efi --model poisson
python train.py --algo pinn_efi --model poisson-inverse
```

Stochastic variant used when noise variance unknown:
```bash
python train.py --algo pinn_efi_sd --model montroll
```

### PINN with dropout (baseline)
```bash
python train.py --algo pinn --model poisson
python train.py --algo pinn --model poisson-inverse
```

### Bayesian PINN (baseline)
```bash
python train.py --algo bpinn --model poisson
python train.py --algo bpinn --model poisson-inverse
```

Outputs (figures, metrics, checkpoints) are saved under `output/<dataset-or-problem>/<algo>/`.

---

## Citation

Until the official NeurIPS proceedings are available, please cite the arXiv preprint (and OpenReview entry if preferred). If you find this repository helpful, please cite:

```bibtex
@misc{shih2025uncertaintyquantificationphysicsinformedneural,
  title         = {Uncertainty Quantification for Physics-Informed Neural Networks with Extended Fiducial Inference},
  author        = {Frank Shih and Zhenghao Jiang and Faming Liang},
  year          = {2025},
  eprint        = {2505.19136},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  doi           = {10.48550/arXiv.2505.19136},
  url           = {https://arxiv.org/abs/2505.19136}
}
```

---

## License

This code is released for research purposes. Please see `LICENSE` if provided, or contact the authors for other uses.

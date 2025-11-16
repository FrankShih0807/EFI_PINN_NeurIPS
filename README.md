# Physics-Informed Neural Networks with Pytorch

requirements are:
- torch
- hamiltorch
- scikit-learn
- numpy
- matplotlib
- seaborn
- ruamel.yaml 
- tqdm

## install package and requirements
```bash
pip install --use-pep517 -e .
pip install -r requirements.txt
```

## run efi pinn
```bash
python train.py --algo pinn_efi --model poisson
python train.py --algo pinn_efi --model poisson-inverse

python train.py --algo pinn_efi_sd --model montroll
```
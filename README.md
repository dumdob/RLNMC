# Reinforcement Learning Nonlocal Monte Carlo (RLNMC)
A JAX implementation of RLNMC for the paper "Nonlocal Monte Carlo via Reinforcement Learning"

## Installation
Install dependencies from requirements.txt (may install the CPU version of JAX, modify for CUDA/TPU, if needed)
```
pip install -r requirements.txt
```
PS. The multi-device use of Gurobi can require a special license. Wandb requires an account.

## Training
```
python experiments/pretrain_nmc.py --config config_train_uf
```
(uses one "device": a GPU recommended)

## Benchmarking
```
python experiments/bench.py --config config_bench_uf
```
(can use many "devices": GPUs recommended)

## Citation
The bibtex citation for the RLNMC paper is:
```
@misc{dobrynin2025nmc,
      title={Nonlocal Monte Carlo via Reinforcement Learning}, 
      author={Dmitrii Dobrynin and Masoud Mohseni and John Paul Strachan},
      year={2025},
      eprint={2508.10520},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.10520}, 
}
```

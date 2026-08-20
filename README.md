# Transversal Gates for Nonadditive Quantum Error-Correcting Codes

Public data and scripts accompanying the paper:

> **Transversal Gates for Nonadditive Quantum Error-Correcting Codes**  
> <https://arxiv.org/abs/2504.20847>

## Overview

This repository contains numerical experiments for searching transversal gates on nonadditive (non-stabilizer) quantum error-correcting codes of the form `((n, 2, 3))`, i.e., single-qubit logical codes encoding 1 qubit into *n* physical qubits with distance 3. The search is formulated as a continuous optimization problem on Riemannian manifolds using PyTorch and the [numqi](https://github.com/numqi/numqi) library.

## Repository Structure

| File / Folder | Description |
|---|---|
| `utils.py` | Core model classes and helper functions (Riemannian optimization models for `((n,2,3))` codes, BD/C-group LP checks) |
| `utils01.py` / `utils02.py` | Additional utility functions used by draft scripts |
| `utils_cws.py` | Utilities related to codeword-stabilized (CWS) codes |
| `draft01_723T.py` | Search for transversal T gate on `((7,2,3))` codes |
| `draft623C10.py` | Search for transversal C10-group gates on `((6,2,3))` codes |
| `draft623_group_range.py` | Range search over group parameters for `((6,2,3))` codes |
| `draft_723_group_range.py` | Range search over group parameters for `((7,2,3))` codes |
| `draft_BDn.py` | Search for transversal BD*n*-group gates |
| `draft_Cn.py` | Search for transversal C*n*-group (cyclic) gates |
| `draft823_BDn.py` | Search for transversal BD*n*-group gates on `((8,2,3))` codes |
| `ws00_convergence/` | Convergence analysis data, plots, and scripts |

## Dependencies

- [numqi](https://github.com/numqi/numqi)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [opt_einsum](https://github.com/dgasmith/opt_einsum)
- [cvxpy](https://www.cvxpy.org/)

## Citation

If you use this code or data, please cite:

```bibtex
@article{transversal2025,
  title   = {Transversal Gates for Nonadditive Quantum Error-Correcting Codes},
  author  = {},
  journal = {arXiv preprint arXiv:2504.20847},
  year    = {2025},
  url     = {https://arxiv.org/abs/2504.20847}
}
```

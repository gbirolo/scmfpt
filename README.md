# scmfpt
Simple convex matrix factorization in pytorch

## Install
Install the package:
```sh
pip install git+https://github.com/gbirolo/scmfpt.git
```

## Usage
To compute components and profiles from a matrix X with individuals as rows and features as columns:
```python
from scmfpt import SCMF
snmf = SCMF(3, n_samples=X.shape[0], n_feats=X.shape[1], epochs=50000)
pred_comps, pred_profs = snmf.fit(X)
```

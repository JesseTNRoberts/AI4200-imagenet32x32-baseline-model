# AI 4200 Project 1: Residual MLP baseline

The instructor baseline. Fork this repo as the starting point for your submission.

## Install

```bash
pip install git+https://github.com/JesseTNRoberts/AI4200-imagenet32x32-baseline-model.git
```

## Files

- `src/mlp_baseline/model.py`: the residual MLP. One class for all three tasks; only the input size and number of classes change.
- `src/mlp_baseline/train.py`: the required `train` and `predict` functions.

## Use

```python
from mlp_baseline import train, predict

model = train("cifar10", X_train, y_train, seed=0, device="cuda")
probs = predict(model, X_test)
accuracy = (probs.argmax(1) == y_test).float().mean()
```

`task` is `"mnist"`, `"cifar10"`, or `"imagenet32"`.

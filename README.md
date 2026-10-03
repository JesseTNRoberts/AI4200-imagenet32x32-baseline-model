# AI 4200 Project 1: Residual MLP baseline

## The assignment

Your job is to build, train, and analyze a residual MLP on three image classification benchmarks: **MNIST**, **CIFAR-10**, and **ImageNet 32×32**. This repo is a deliberately plain starting point. It is meant to be beaten.

**What you can change.** You can change anything about how the network is built and trained, as long as it stays an MLP made from these four pieces:

- **Linear layers.** Any depth and any width.
- **Activations.** ReLU, tanh, or anything else.
- **Normalization.** Batch norm, layer norm, or anything else.
- **Residual connections.**

You can also experiment freely with:

- **Training.** Loss function, optimizer, learning-rate schedule, regularization (dropout, weight decay, and so on), and initialization.
- **Data handling.** Data augmentation, test-time augmentation, and creative ways of feeding images into an MLP, such as splitting an image into patches or mixing information across positions. You must be able to explain any of these in your report.

**What you cannot do.**

- **No new layer types.** No convolutions, recurrent layers, or other modality-specific layers.
- **No ensembles.** Each submission is one model.
- **No pretrained weights and no distillation.** Your model must be trained from the data directly.
- **No extra data.** Use only the data each benchmark provides.
- **No saved weights.** Your code must build and train the model from scratch when `train()` is called. It may not load saved weights.

**Document every run.** The core of the grade is your development process. For every training run, record:

- **What you changed** and why.
- **What happened**: the accuracy and the loss curves.
- **Your reading of the result.**
- **What you will try next**, based on that reading.

A run that fails but is well analyzed is worth more than a lucky one that isn't explained.

**How you submit.** Submit a public GitHub repository that installs with:

```bash
pip install git+https://github.com/<you>/<your-repo>.git
```

Your package must expose the two functions in `train.py`, with their names and arguments unchanged:

```python
model = train(task, X_train, y_train, seed=0, device="cuda")   # build and train from scratch
probs = predict(model, X_test)                                 # (N, num_classes) probabilities
```

The course notebook downloads the data, installs your package, calls these two functions for each task, and reports test accuracy. Do your development in your own environment. The notebook is only the test harness.

**Leaderboard.** Test accuracies will be compared "leaderboard" style. The leaderboard does not count toward your grade. The single top-performing model earns extra credit: **3 points** if its author is in the 4000-level section, or **2 points** if in the 5000-level section.

## Install this baseline model

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

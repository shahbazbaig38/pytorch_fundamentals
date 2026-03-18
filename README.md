# Agentic AI — PyTorch Fundamentals Demo

Small collection of PyTorch example scripts aimed at demonstrating core concepts from tensors through training a simple neural network.

---

## 📌 Project Structure

- `pyproject.toml` — project metadata and dependencies
- `README.md` — this file
- `pytorch_fundamentals/` — example scripts demonstrating PyTorch basics:
  - `tensors.py` — tensor shapes, dtypes, and basic creation
  - `tensor_operations.py` — basic tensor arithmetic and matrix multiplication
  - `computation_graph.py` — forward computation for a logistic-like model
  - `autograd.py` — gradient computation using `backward()` and `torch.autograd.grad`
  - `data_loader.py` — synthetic dataset + `Dataset`/`DataLoader` usage
  - `neural_network.py` — a simple `torch.nn.Module` network + forward pass
  - `training_loop.py` — training loop, evaluation, checkpoint save/load
  - `model.pth` — saved weights from the training example

---

## 🔧 Requirements

- Python **>= 3.13** (as declared in `pyproject.toml`)
- Dependencies (installed via `pip`):
  - `torch` (PyTorch)
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `plotly`

---

## 🚀 Quick Start

1) Create and activate a virtual environment (recommended):


2) Install dependencies:

> Note: This repo currently uses `pyproject.toml` for dependency metadata but does not ship with a `requirements.txt`. You can generate one with `pip freeze > requirements.txt` if you want.

---

## ▶️ Running the examples

Run any of the example scripts from the repository root:

```bash
python pytorch_fundamentals/tensors.py
python pytorch_fundamentals/tensor_operations.py
python pytorch_fundamentals/computation_graph.py
python pytorch_fundamentals/autograd.py
python pytorch_fundamentals/data_loader.py
python pytorch_fundamentals/neural_network.py
python pytorch_fundamentals/training_loop.py
```

Each script prints basic output to the console explaining what it demonstrates.

---

## 🧠 What’s Demonstrated

- **Tensor basics** (`tensors.py`) — shapes, dimensions, dtype conversions
- **Basic ops** (`tensor_operations.py`) — matrix multiplication and tensor math
- **Forward computation** (`computation_graph.py`) — building a computation graph through operations
- **Autograd / gradients** (`autograd.py`) — differentiating a loss w.r.t. parameters
- **Data loading** (`data_loader.py`) — building `Dataset` + `DataLoader` and iterating batches
- **Model definition** (`neural_network.py`) — `torch.nn.Module` and `Sequential`
- **Training loop** (`training_loop.py`) — optimization loop, loss computation, evaluation, checkpointing

---

## 🔁 Persisted Model

The training example writes a checkpoint:
- `pytorch_fundamentals/model.pth`

You can load it back with the same model class (see `training_loop.py`).

---

## 📌 Notes / Next Steps

- The code is intended as a learning/demo repo. Feel free to extend the dataset, add metrics, or build a training/validation pipeline.
- If you want a runnable CLI, consider adding an entrypoint script (e.g., `run.py`) that selects which demo to run.

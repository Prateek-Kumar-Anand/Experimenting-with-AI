# 🧮 Data Handling in Form of Tensors

This folder covers **PyTorch tensor fundamentals** — the building block of all deep learning models. Three progressive notebooks take you from basic arithmetic all the way to matrix operations and shape manipulation.

---

## 📓 Notebooks

### 1. `basic operation on tensor.ipynb`
Introduction to creating and doing arithmetic with 1D tensors.

**What's covered:**
- Creating tensors from lists using `torch.tensor()`
- Element-wise addition, subtraction, multiplication
- Printing and inspecting tensor values

**Key code snippet:**
```python
import torch
a = torch.tensor([2, 3, 4, 5])
b = torch.tensor([5, 4, 3, 2])
c = a + b   # tensor([7, 7, 7, 7])
e = a * b   # tensor([10, 12, 12, 10])
```

---

### 2. `named tensor.ipynb`
Working with multi-dimensional tensors and understanding how dimensions map to real data (channels, rows, columns, batches). Also includes GPU verification.

**What's covered:**
- CUDA availability check (`torch.cuda.is_available()`)
- 3D tensors — shape `(channels, rows, columns)` using `torch.randn()`
- 4D tensors — shape `(batch, channels, rows, columns)`
- Understanding that `torch.randn()` produces different values each run

**Key code snippet:**
```python
# 3D tensor — 3 channels, 5x5 image
img_t = torch.randn(3, 5, 5)

# 4D tensor — 1 batch, 2 channels, 2x1
a = torch.randn(1, 2, 2, 1)

# torch.randn(batch, channel, row, column)
```

---

### 3. `advanced operations on tensor.ipynb`
Heavier tensor operations — matrix math, dot products, reshaping, and reduction functions used in real ML workflows.

**What's covered:**
- Matrix multiplication with `torch.matmul()`
- Flattening 2D tensors with `torch.flatten()` for dot product
- Dot product using `torch.dot()`
- Reshaping tensors with `.reshape()`
- Finding max index with `torch.argmax()` (useful for predictions)
- Computing mean with `torch.mean()` (useful for loss)

**Key code snippet:**
```python
# Matrix multiplication
result = torch.matmul(a, b)

# Dot product (need 1D tensors)
A1 = torch.flatten(a)
B1 = torch.flatten(b)
dot = torch.dot(A1, B1)   # tensor(26.4600)

# Reshape
test = torch.tensor([4, 4, 2, 5])
c = test.reshape(2, 2)   # 2x2 matrix
d = test.reshape(4, 1)   # column vector

# Argmax — returns index of max value
torch.argmax(test)   # tensor(3)  → index 3 has value 5

# Mean — useful as a simple loss metric
torch.mean(test.float())   # tensor(3.7500)
```

---

## 🧠 Concepts Summary

| Concept | Function | Use case |
|---------|----------|----------|
| Create tensor | `torch.tensor()` | From known values |
| Random tensor | `torch.randn()` / `torch.rand()` | Dummy data, weight init |
| Matrix multiply | `torch.matmul()` | Linear layers |
| Dot product | `torch.dot()` | Similarity, projections |
| Reshape | `.reshape()` | Changing tensor dimensions |
| Flatten | `torch.flatten()` | Prep for FC layers |
| Max index | `torch.argmax()` | Classification output |
| Mean | `torch.mean()` | Loss calculation |

---

## ⚙️ Environment

All notebooks run on **Google Colab** with a **Tesla T4 GPU**.

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # Tesla T4
```

---

## ▶️ Running the Notebooks

Open in Google Colab (recommended):

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. File → Open Notebook → GitHub → paste repo URL
3. Select any `.ipynb` and run cells top to bottom

Or locally:
```bash
pip install torch jupyter
jupyter notebook
```

---

## 📌 Notes

- `torch.randn()` generates values from a standard normal distribution — different every run.
- `.reshape()` does **not** copy data; it creates a new view of the same memory.
- `torch.argmax()` returns the **index** of the max value, not the value itself.
- Always cast to `.float()` before using `torch.mean()` on integer tensors.

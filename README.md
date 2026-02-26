# 🧠 Python Numerical Projects (NumPy-Based)

A collection of applied numerical computing and scientific simulation projects implemented using **NumPy**, **SciPy**, and **Matplotlib**.

This repository demonstrates practical implementations of:
- Numerical methods
- Optimization techniques
- Signal & motion analysis
- Clustering algorithms
- Geometric computations
- Implicit ODE solvers
- Data-driven simulations

All projects are written in Python and rely primarily on **NumPy for vectorized computation**.

---

## 🔬 Projects Overview

### 1️⃣ Battery Thermal Runaway – Implicit Solvers
📂 `battery-thermal-runaway-implicit-solvers`

Simulation of a nonlinear ODE system modeling battery thermal runaway using:

- Implicit Euler method
- Newton–Gauss–Seidel iterative solver
- Stability comparison with explicit schemes
- Temperature growth visualization

**Concepts used:**
- Initial Value Problems (IVP)
- Nonlinear system solving
- Iterative numerical methods
- Matrix-free Jacobian approximation

---

### 2️⃣ Candle Volume Reconstruction
📂 `candle-volume-reconstruction`

Reconstruction and analysis of financial candle volume data.

**Concepts used:**
- Time-series processing
- Vectorized NumPy transformations
- Numerical aggregation techniques
- Data smoothing

---

### 3️⃣ Energy Load Clustering
📂 `energy-load-clustering`

Clustering and classification of energy consumption patterns.

**Algorithms implemented:**
- K-Means (NumPy-based)
- Distance matrix computation
- Feature normalization
- Cluster visualization

---

### 4️⃣ Finite Difference Analysis
📂 `finite-difference-analysis`

Numerical differentiation and PDE-style discretization using:

- Forward / Backward difference
- Central difference schemes
- Error analysis
- Convergence behavior

---

### 5️⃣ Norm Visualizer
📂 `norm-visualizer`

Visualization and comparison of different vector norms:

- L1 norm
- L2 norm
- L∞ norm
- General p-norms

Includes geometric interpretation and contour visualization.

---

### 6️⃣ Spline Curve Analysis
📂 `spline-curve-analysis`

Implementation and visualization of:

- Parametric splines
- Interpolation techniques
- Curve smoothness analysis
- NumPy-based matrix formulations

---

### 7️⃣ Video Motion Analysis & Clustering
📂 `video-motion-analysis-clustering`

Motion vector extraction and clustering from video frames.

**Concepts used:**
- Vector field processing
- Optical flow data handling
- Motion clustering
- NumPy-based spatial filtering

---

### 8️⃣ Voronoi Norm Comparison
📂 `voronoi-norm-comparison`

Comparison of Voronoi diagrams under different distance metrics:

- Euclidean (L2)
- Manhattan (L1)
- Chebyshev (L∞)
- Custom p-norms

Demonstrates how metric choice changes spatial partitioning.

---

## 🛠 Technologies Used

- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- OpenCV (for motion project)

---

## 🚀 Installation

```bash
git clone https://github.com/levanilekvinadze24/python-projects.git
cd python-projects
pip install -r requirements.txt

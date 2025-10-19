# Machine Learning with scikit-learn and pandas

Includes machine learning classifier and regressors.

## Computer_Vision_in_Agriculture

This project demonstrates a simple **K-Nearest Neighbors (KNN)** classifier applied to a **fruit dataset**.
It shows how to use KNN for supervised classification, visualize decision boundaries, and evaluate model performance.

The project is based on an educational example from the *Applied Machine Learning with Python* course (Coursera), extended with clear plots and explanations.

---

### Jupyter Notebook

For a full walkthrough with code, outputs, and visualizations, see the
Jupyter Notebook [Computer_Vision_in_Agriculture.ipynb](Computer_Vision_in_Agriculture.ipynb)

Run the notebook online (no setup required):
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MahbubAlam231/machine-learning-scikit-learn-pandas/main?filepath=Computer_Vision_in_Agriculture.ipynb)

---

### Project Overview

- **Goal:** Classify fruits based on features (weight, height, width, color score, etc.)
- **Algorithm:** K-Nearest Neighbors (KNN)
- **Steps:**
  1. Load the fruit dataset
  2. Split into training and test sets
  3. Standardize features
  4. Train a KNN classifier with different `k` values
  5. Visualize classification accuracy and decision boundaries
  6. Evaluate the model

---

### Dataset

The dataset contains fruit samples with the following attributes:

- Fruit label (apple, mandarin, orange, lemon)
- Features: height, width, mass, color score

"""
#!/usr/bin/python3
 Author(s)   : Mahbub Alam
 File        : Computer_Vision_in_Agriculture.py
 Created     : 2025-04-04 (Apr, Fri) 13:39:36 CEST
 Description : k-nearest neighbor classifier.# {{{
A small script that trains and evaluates a k-nearest neighbors classifier for
fruit detection.

# Purpose
-------
- Load features/labels for fruit images.
- Train a KNN classifier and evaluate accuracy/metrics.
- Predict on new images.

# }}}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # importing classifier constructor
import warnings
warnings.filterwarnings("ignore", message="Creating legend with loc=")

fruits = pd.read_table('fruit_data_with_colors.txt')

print(fruits.head())

# print(fruits.columns) # Output: ['fruit_label', 'fruit_name', 'fruit_subtype', 'mass', 'width', 'height', 'color_score']

print(f"")

# ===============[[ check missing data ]]================{{{
print(f"")
print(68*"=")
print(f"==={19*'='}[[ check missing data ]]{19*'='}===\n")
# ==========================================================

missing_data = fruits.isna().any()  # returns bool on columns
cols_with_nan = fruits.columns[missing_data].to_list()
print(f"")

# }}}

# =========[[ fruit label and name dictionary ]]========={{{
print(f"")
print(68*"=")
print(f"==={13*'='}[[ fruit label and name dictionary ]]{13*'='}==\n")
# ==========================================================

lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
# print(lookup_fruit_name) # Output: {1: 'apple', 2: 'mandarin', 3: 'orange', 4: 'lemon'}

# }}}

# ================[[ train test split ]]================={{{
print(f"")
print(68*"=")
print(f"==={20*'='}[[ train test split ]]{20*'='}===\n")
# ==========================================================

X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# }}}

# ==========[[ learning with KNN classifier ]]==========={{{
print(f"")
print(68*"=")
print(f"==={14*'='}[[ learning with KNN classifier ]]{14*'='}===\n")
# ==========================================================

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, y_train)

print(f"Accuracy score: {knn.score(X_test, y_test)}")


example_fruits = pd.DataFrame([[20, 4.3, 5.1], [180, 7.8, 8.3]], columns = X.columns)
fruit_predictions = knn.predict(example_fruits)
print([lookup_fruit_name[label] for label in fruit_predictions])

# }}}

# ======[[ decision boudaries for KNN classifier ]]======{{{
print(f"")
print(68*"=")
print(f"==={10*'='}[[ decision boudaries for KNN classifier ]]{10*'='}==\n")
# ==========================================================

from utils import plot_fruit_knn

plot_fruit_knn(X_train, y_train, 5, 'uniform')

# }}}

# ============[[ classifier accuracy vs k ]]============={{{
print(f"")
print(68*"=")
print(f"==={16*'='}[[ classifier accuracy vs k ]]{16*'='}===\n")
# ==========================================================

k_range = range(1, 20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.scatter(k_range, scores)
# Axis labels and title
plt.xlabel("k")
plt.ylabel("accuracy")
plt.xticks(range(0, 20, 5))
plt.savefig('knn_accuracy_vs_k.jpg')
plt.show()

# }}}

# =====[[ classifier accuracy vs train/test split ]]====={{{
print(f"")
print(68*"=")
print(f"==={9*'='}[[ classifier accuracy vs train/test split ]]{9*'='}==\n")
# ==========================================================

t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 5)

file_name = 'knn_accuracy_vs_train_test_split'

train_props = []
avg_scores = []

for s in t:
    scores = []
    for i in range(1, 1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))

    train_props.append(s * 100)  # Convert to percentage
    avg_scores.append(np.mean(scores))

plt.figure(figsize=(8, 6))
plt.plot(train_props, avg_scores, 'bo-')  # Line with points
plt.xlabel('Training set proportion (%)')
plt.ylabel('Accuracy')
plt.title(file_name)
plt.grid(True)
plt.savefig(f'{file_name}.jpg')
plt.show()

# }}}


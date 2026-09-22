# Principal Component Analysis (PCA) with PySpark

## Based on What I Learned and Applied

This documentation continues from the **Feature Engineering** process.

The previous stage produced the standardized `features_scaled` vector, which I used as the input for PCA.

To make the concept easier to understand, I first continue with the **sushi example** from Feature Engineering and then show the **PCA implementation and results from my thesis**.

> **Learning note**
>
> This repository documents what I implemented and learned while working with Databricks, PySpark, and Spark ML.
>
> The sushi examples are simplified learning examples. The thesis sections show the implementation and results from my academic project.

The complete flow is:

```text
Feature Engineering
        ↓
StandardScaler
        ↓
features_scaled
        ↓
PCA
        ↓
pca_features
        ↓
K-Means
```

---

# What I Learned

## 1. What is PCA and Why Did I Use It?

PCA stands for **Principal Component Analysis**.

It is a dimensionality reduction technique.

A simple way I understand it is:

```text
Many Features
      ↓
     PCA
      ↓
Fewer Dimensions
```

Having several dimensions does not automatically mean that PCA is required.

For example, K-Means can work directly with standardized features:

```text
features_scaled
       ↓
    K-Means
```

In my ML workflow, I used PCA to explore whether the original feature space could be represented using fewer dimensions before applying K-Means.

---

##  2. Simple Example — Sushi Customers

I can continue with the sushi restaurant example from Feature Engineering.

Imagine that each customer is described by:

```text
Age
Orders per Month
Average Order Price
Monthly Spending
Sushi Orders
Other Food Orders
Uses Promotions
Orders on Weekends
```

That means:

```text
8 Features
    =
8 Dimensions
```

A customer could conceptually be represented as:

```text
Customer A

Age                 = 30
Orders per Month    = 12
Average Order Price = 42
Monthly Spending    = 504
Sushi Orders        = 8
Other Food Orders   = 4
Uses Promotions     = 1
Orders on Weekends  = 1
```

After Feature Engineering and scaling, these values would be represented in:

```text
features_scaled
```

PCA can create a smaller representation:

```text
8 Original Features
        ↓
       PCA
        ↓
   PC1  PC2  PC3
```

Instead of representing each sushi customer using 8 dimensions, the customer can now be represented using three Principal Components:

```text
Customer A

[PC1, PC2, PC3]
```

The important idea is:

```text
Original Customer

[f1, f2, f3, f4, f5, f6, f7, f8]

                  ↓ PCA

Reduced Representation

[PC1, PC2, PC3]
```

PCA does not simply delete five columns.

It creates new components using information from the original features.

---

## 3. PCA Does Not Select the "Best" Features

One important thing I learned is that PCA does **not** select the best original features.

It does not do this:

```text
8 Features
    ↓
Select the best 3
    ↓
Age
Monthly Spending
Sushi Orders
```

Instead:

```text
8 Original Features
        ↓
       PCA
        ↓
PC1   PC2   PC3
```

The Principal Components are mathematically constructed using information from the original features.

Therefore:

```text
Feature Selection
       ↓
Keeps original features


PCA
       ↓
Creates new components
```

For example, conceptually a Principal Component could combine information from several original variables:

```text
PC1
 │
 ├── information from Orders
 ├── information from Spending
 ├── information from Sushi Orders
 └── information from other features
```

This does **not** mean that PC1 literally represents "spending" or "orders".

The meaning of a Principal Component depends on how the original features contribute to it.

---

## 4. PCA Does Not Create Groups

Another important distinction I learned is that Principal Components are **not clusters**.

```text
PCA
 ↓
Reduces dimensions


K-Means
 ↓
Creates groups of similar records
```

Using the sushi example:

```text
Sushi Customer Features
        ↓
       PCA
        ↓
[PC1, PC2, PC3]
```

At this point, the customers have a new numerical representation.

They have **not yet been grouped**.

Later:

```text
[PC1, PC2, PC3]
        ↓
     K-Means
        ↓
Customer Groups
```

PCA changes how the records are represented.

K-Means uses that representation to search for groups of similar records.

---

## 5. How Many Principal Components Did I Use?

In PySpark, the number of Principal Components is defined using `k`.

For example:

```python
PCA(k=3)
```

means:

```text
Create 3 Principal Components

        ↓

PC1   PC2   PC3
```

An important distinction I learned is:

```text
I choose k
    ↓
PCA learns how the components
should be constructed
```

Therefore:

```python
k = 3
```

does **not** mean that PCA automatically determined that three components were the optimal number.

To understand how much information those components represent, I examined the **explained variance**.

---

## 6. Understanding Explained Variance

Suppose the sushi example produced:

```text
PC1 → 35%
PC2 → 22%
PC3 → 15%
```

Each percentage represents how much of the variability in the original feature space is represented by that component.

Together:

```text
35% + 22% + 15% = 72%
```

Therefore:

```text
8 Original Features
        ↓
       PCA
        ↓
PC1 + PC2 + PC3
        ↓
72% Cumulative Explained Variance
```

This means that the three Principal Components represent approximately **72% of the variability contained in the original features**.

It does **not** mean:

```text
PCA is 72% accurate
```

and it does **not** mean:

```text
72% of the original columns were preserved
```

The representation changed from:

```text
8 Dimensions
     ↓
3 Dimensions
```

and those three dimensions represent 72% of the original variability.

This helped me understand the trade-off:

```text
Fewer Dimensions
       ↓
Smaller Representation
       ↓
Some Original Variability
May Be Lost
```

---

# PCA in My Thesis with PySpark and Databricks

## 7. Loading the Prepared Features

For my thesis implementation, I started from the standardized features created during the previous Feature Engineering stage.

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
import numpy as np
import matplotlib.pyplot as plt

spark = SparkSession.builder.getOrCreate()

df_features = spark.table(
    "mart_parqueo.ml_features_scaled"
)
```

I inspected the PCA input:

```python
df_features.select(
    "features_scaled"
).show(
    3,
    truncate=False
)
```

The input for PCA was:

```text
features_scaled
```

The workflow at this point was:

```text
Thesis Data
      ↓
Feature Engineering
      ↓
VectorAssembler
      ↓
features_raw
      ↓
StandardScaler
      ↓
features_scaled
      ↓
PCA
```

---

## 8. Applying PCA

I configured PCA with three Principal Components:

```python
pca = PCA(
    k=3,
    inputCol="features_scaled",
    outputCol="pca_features"
)
```

The main parameters are:

| Parameter | Purpose |
|---|---|
| `k` | Number of Principal Components |
| `inputCol` | Input feature vector |
| `outputCol` | PCA output vector |

Conceptually:

```text
features_scaled
[f1, f2, ... fn]

        ↓

     PCA(k=3)

        ↓

pca_features
[PC1, PC2, PC3]
```

---

## 9. Understanding `fit()` and `transform()`

Another concept I learned during the implementation was the difference between `fit()` and `transform()`.

A simple way I understand these operations is:

```text
fit()
 ↓
Learn from the data


transform()
 ↓
Apply what was learned
```

First, PCA learned how the Principal Components should be constructed:

```python
pca_model = pca.fit(
    df_features
)
```

Then I applied that learned transformation to the records:

```python
df_pca = pca_model.transform(
    df_features
)
```

So:

```text
features_scaled
       ↓
      fit()
       ↓
PCA learns the transformation
       ↓
   transform()
       ↓
pca_features
```

Not every PySpark transformation needs `fit()`.

For example, `VectorAssembler` can use `transform()` directly because it does not need to learn parameters from the dataset.

This distinction helped me understand the difference between a Spark ML **Transformer** and an **Estimator** that needs to learn from data.

---

## 10. Explained Variance in PySpark

After fitting PCA, I obtained the explained variance:

```python
explained_variance = (
    pca_model.explainedVariance
)
```

To display the variance represented by each component:

```python
for i, variance in enumerate(
    explained_variance,
    start=1
):
    print(
        f"PC{i}: "
        f"{variance:.4f} "
        f"({variance * 100:.2f}%)"
    )
```

To calculate the cumulative explained variance:

```python
cumulative_variance = np.cumsum(
    explained_variance
)
```

Then:

```python
for i, variance in enumerate(
    cumulative_variance,
    start=1
):
    print(
        f"PC1-PC{i}: "
        f"{variance:.4f} "
        f"({variance * 100:.2f}%)"
    )
```

---

## 11. My Practical Result

In my implementation, I started with **16 features** and configured:

```python
k = 3
```

The explained variance was approximately:

```text
PC1 → 20.48%
PC2 → 15.06%
PC3 → 13.00%
```

The cumulative result was:

```text
PC1             → 20.48%
PC1 + PC2       → 35.54%
PC1 + PC2 + PC3 → 48.54%
```

Therefore:

```text
16 Original Features
        ↓
    PCA (k=3)
        ↓
3 Principal Components
        ↓
48.54% Cumulative
Explained Variance
```

This means that the three Principal Components represented approximately **48.54% of the variability contained in the original feature set**.

It does **not** mean that PCA was 48.54% accurate.

It also does **not** mean that:

```python
k = 3
```

was automatically the optimal number of components.

It describes how much of the original variability was represented by the three components I selected for this exercise.

This was one of the most important differences I learned:

```text
k = 3
       ≠
PCA decided that 3 is optimal
```

---

## 12. Sushi Example vs. My Thesis

At this point, the relationship between the learning example and my implementation can be summarized as:

```text
 SUSHI EXAMPLE

8 Features
    ↓
StandardScaler
    ↓
features_scaled
    ↓
PCA
    ↓
3 Principal Components
    ↓
Example: 72% Variance


 MY THESIS

16 Features
    ↓
StandardScaler
    ↓
features_scaled
    ↓
PCA (k=3)
    ↓
3 Principal Components
    ↓
48.54% Cumulative Explained Variance
```

The sushi numbers are only a simplified example used to understand the concept.

The **48.54% result corresponds to my PCA implementation in the thesis exercise**.

---

## 13. Visualizing Explained Variance

I visualized the result using Matplotlib:

```python
variance_pct = explained_variance * 100

components = [
    f"PC{i}"
    for i in range(
        1,
        len(variance_pct) + 1
    )
]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    components,
    variance_pct
)

plt.xlabel("Principal Components")
plt.ylabel("Explained Variance (%)")
plt.title(
    "Explained Variance by Principal Component"
)

for bar, value in zip(
    bars,
    variance_pct
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.4,
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()
```

This visualization made it easier for me to compare how much variance each Principal Component represented.

---

## 14. PCA with Python / Scikit-learn

PCA is the technique, while PySpark and Scikit-learn are different tools that can be used to implement it.

With Scikit-learn:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_scaled)
```

In this case:

```text
PySpark
k = 3


Scikit-learn
n_components = 3
```

Both configurations create three Principal Components.

The explained variance can be obtained with:

```python
pca.explained_variance_ratio_
```

Scikit-learn also allows a variance target to be specified:

```python
pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)
```

Conceptually:

```text
Preserve approximately
95% of the variance
        ↓
PCA determines how many
components are needed
```

This helped me understand that the PCA concept is independent of the library used to implement it.

---

## 15. Saving the PCA Result in Databricks

After applying PCA, I saved the reduced representation as a Delta table.

```python
df_pca.select(
    "id_encuesta",
    "pca_features"
).write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "mart_parqueo.ml_features_pca"
    )
```

The result of this stage was:

```text
mart_parqueo.ml_features_pca
```

I used this approach because I separated the Machine Learning process into stages:

```text
Feature Engineering
        ↓
mart_parqueo.ml_features_scaled
        ↓
PCA
        ↓
mart_parqueo.ml_features_pca
        ↓
K-Means
```

Saving the PCA result was not required by the PCA algorithm itself.

It was part of how I organized my learning workflow so I could inspect and validate each stage separately.

---

## 16. What Did PCA Produce?

Before PCA, each record was represented using the original feature space:

```text
features_scaled
```

After PCA, each record had a new representation:

```text
pca_features
```

containing:

```text
[PC1, PC2, PC3]
```

For example, one output vector could look like:

```text
[-1.71, 0.02, -0.01]
```

These numbers are not:

```text
Age
Price
Location
Profile
```

They are coordinates in the new PCA feature space.

That distinction was important for me:

```text
Original Features
      ↓
      PCA
      ↓
New Mathematical Representation
      ↓
[PC1, PC2, PC3]
```

---

# 17. PCA vs. K-Means

After PCA, the records were ready for the clustering stage.

```text
Original Features
       ↓
StandardScaler
       ↓
features_scaled
       ↓
PCA
       ↓
[PC1, PC2, PC3]
       ↓
K-Means
       ↓
Groups of Similar Records
```

The difference I needed to remember was:

```text
PCA
 ↓
Can I represent my feature space
with fewer dimensions?


K-Means
 ↓
Which records are similar
to each other?
```

PCA changes the representation.

K-Means creates the groups.

---

# What I Learned

Through this implementation I learned that:

- **PCA reduces dimensionality** by creating new Principal Components.
- PCA does **not** simply select the best original features.
- PCA does **not** create clusters.
- `k` defines how many Principal Components I choose to create.
- PCA learns how those components are constructed from the data.
- `fit()` learns the PCA transformation and `transform()` applies it.
- **Explained variance** indicates how much of the original variability is represented by the components.
- Explained variance is **not model accuracy**.
- Reducing dimensions involves a trade-off because some original variability can be lost.
- PCA is **not mandatory before K-Means**.
- In my implementation, I used the PCA representation as the input for K-Means.
- The output values in `pca_features` are new coordinates, not the original variables.

My complete PCA process was:

```text
Feature Engineering
        ↓
StandardScaler
        ↓
16 Features
        ↓
PCA (k=3)
        ↓
3 Principal Components
        ↓
48.54% Cumulative Explained Variance
        ↓
mart_parqueo.ml_features_pca
        ↓
K-Means
```

---

# Important Note

This document describes a **learning implementation based on my academic project**.

The sushi example is used only to make the PCA concepts easier to understand.

The PCA configuration and the **48.54% cumulative explained variance** correspond to the implementation documented in my thesis exercise.

The use of:

```python
k = 3
```

should not be interpreted as a universal recommendation.

The appropriate number of Principal Components depends on the dataset, the amount of variance that needs to be retained, the objective of the analysis, and the effect of dimensionality reduction on subsequent models.

---

# Next Stage — K-Means

After PCA, each record in my implementation was represented using:

```text
[PC1, PC2, PC3]
```

These PCA features became the input for the clustering stage:

```text
pca_features
      ↓
K-Means
      ↓
Clusters
      ↓
Interpretation
```

The next document explains how I implemented and evaluated that clustering process.

➡️ [Machine Learning — K-Means](Machine_Learning_K-Means.md)

---

⬅️ [Back to main README](README.md)

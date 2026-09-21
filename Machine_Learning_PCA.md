# Principal Component Analysis (PCA) with PySpark

## Based on What I Learned and Applied

This documentation continues from the **Feature Engineering** process.

The previous stage produced the standardized `features_scaled` vector, which I use as the input for PCA.

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

## What is PCA and Why Would I Use It?

PCA stands for **Principal Component Analysis**.

It is a dimensionality reduction technique.

To understand why I would use it, I can continue with the sushi restaurant example from Feature Engineering.

Suppose I have these customer features:

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

In this example:

```text
8 Features
    =
8 Dimensions
```

Having several dimensions does not automatically mean that I need PCA.

For example, K-Means could work directly with my standardized features:

```text
features_scaled
       ↓
K-Means
```

However, I may want to investigate whether I can represent the same feature space using fewer dimensions.

```text
8 Features
    ↓
   PCA
    ↓
PC1  PC2  PC3
```

Instead of representing each customer with 8 dimensions, PCA creates a smaller representation using **Principal Components**.

A simple way I understand it is:

```text
Many Features
      ↓
     PCA
      ↓
Fewer Dimensions
```

---

## PCA Does Not Select Features or Create Groups

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

Instead, it creates new variables:

```text
8 Original Features
        ↓
       PCA
        ↓
PC1  PC2  PC3
```

The Principal Components are mathematically created using information from the original features.

Therefore:

```text
Feature Selection
       ↓
Keeps original features


PCA
       ↓
Creates new components
```

The Principal Components are also **not clusters**.

```text
PCA
 ↓
Reduces dimensions


K-Means
 ↓
Creates groups of similar records
```

PCA changes how the records are represented. K-Means will later use that representation to find groups.

---

## How Many Principal Components Should I Use?

In PySpark, I choose the number of Principal Components using `k`.

For example:

```python
PCA(k=3)
```

means:

```text
Create 3 Principal Components
        ↓
PC1  PC2  PC3
```

An important distinction is:

```text
I choose k
    ↓
PCA learns how the components
should be constructed
```

Therefore, `k=3` does not mean that PCA automatically determined that three components were the optimal number.

To evaluate my choice, I can look at the **explained variance**.

---

## Understanding Explained Variance

Suppose that in the sushi example I obtain:

```text
PC1 → 35%
PC2 → 22%
PC3 → 15%
```

Each percentage represents how much of the original variability is represented by that component.

Together:

```text
35% + 22% + 15% = 72%
```

So:

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

It does not mean:

```text
PCA is 72% accurate
```

and it does not mean:

```text
72% of the original features were preserved
```

It means that I reduced the representation:

```text
8 Dimensions
     ↓
3 Dimensions
```

and those three dimensions represent 72% of the original variability.

This is the trade-off when using PCA:

```text
Fewer Dimensions
       ↓
Smaller Representation
       ↓
Some Original Variability
May Be Lost
```

---

# PCA with PySpark and Databricks

## Loading the Prepared Features

I start from the standardized features created during Feature Engineering.

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
import numpy as np
import matplotlib.pyplot as plt

spark = SparkSession.builder.getOrCreate()

df_features = spark.table(
    "mart_myproject.ml_features"
)
```

I can inspect the input:

```python
df_features.select(
    "features_scaled"
).show(
    3,
    truncate=False
)
```

The input for PCA is:

```text
features_scaled
```

---

## Applying PCA

I configure PCA with three Principal Components:

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
[f1, f2, ... f16]

        ↓
     PCA(k=3)

        ↓

pca_features
[PC1, PC2, PC3]
```

---

## `fit()` and `transform()`

A simple way I understand these operations is:

```text
fit()
 ↓
Learn from the data


transform()
 ↓
Apply what was learned
```

First, PCA learns how the Principal Components should be constructed:

```python
pca_model = pca.fit(
    df_features
)
```

Then I apply that transformation to the records:

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

Not every PySpark operation needs `fit()`.

For example, `VectorAssembler` can use `transform()` directly because it does not need to learn parameters from the data.

---

## Explained Variance in PySpark

After fitting PCA, I can obtain the explained variance:

```python
explained_variance = (
    pca_model.explainedVariance
)
```

To display each component:

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

## My Practical Result

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
48.54% Cumulative Explained Variance
```

This means that the three Principal Components represented approximately **48.54% of the variability contained in the original feature set**.

It does not mean that PCA was 48.54% accurate.

It also does not mean that `k=3` was automatically the optimal number of components.

It describes how much of the original variability was represented by the three components I chose.

---

## Visualizing Explained Variance

I can visualize the result using Matplotlib:

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

---

# PCA with Python / Scikit-learn

PCA is the technique, while PySpark and Scikit-learn are tools that can be used to implement it.

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

Both create three Principal Components.

The explained variance can be obtained with:

```python
pca.explained_variance_ratio_
```

Scikit-learn also allows me to specify how much variance I want to
preserve:

```python
pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)
```

In this case:

```text
Preserve approximately
95% of the variance
        ↓
PCA determines how many
components are needed
```

---

# Saving the PCA Result in Databricks

I can save the PCA representation as a Delta table:

```python
df_pca.select(
    "record_id",
    "pca_features"
).write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "mart_myproject.ml_features_pca"
    )
```

I used this approach because I separated the Machine Learning process into stages:

```text
Feature Engineering
       ↓
ml_features
       ↓
PCA
       ↓
ml_features_pca
       ↓
K-Means
```

---

# Next Step: K-Means

After PCA, each record is represented using:

```text
[PC1, PC2, PC3]
```

These components become the input for K-Means:

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

The difference I need to remember is:

```text
PCA
 ↓
Can I represent my features with fewer dimensions?


K-Means
 ↓
Which records are similar to each other?
```

---

# What I Learned

- **PCA reduces dimensionality** by creating new Principal Components.
- PCA does **not** select the best original features.
- PCA does **not** create clusters.
- `k` defines how many Principal Components I want to create.
- **Explained variance** tells me how much variability is represented by the components.
- PCA is **not mandatory before K-Means**; in my implementation, I used the PCA representation as the input for clustering.

My complete process was:

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
K-Means
```

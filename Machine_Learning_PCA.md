# Principal Component Analysis (PCA) with PySpark

## Based on What I Learned and Applied

This documentation is based on what I learned about Principal Component
Analysis (PCA) and how I applied those concepts using PySpark and Databricks.

My goal is to document PCA in a simple and practical way: what it is,
how Principal Components work, how to apply PCA, and how to evaluate the
result using explained variance.

This document continues from the Feature Engineering process:

```text
Feature Engineering
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

## What is PCA?

PCA stands for **Principal Component Analysis**.

It is a dimensionality reduction technique used when we have multiple
features and want to represent them using fewer dimensions.

For example, imagine that after Feature Engineering I have:

```text
16 Features

[f1, f2, f3, f4, ... f16]
```

PCA can create a smaller representation:

```text
16 Features
     ↓
    PCA
     ↓
PC1  PC2  PC3
```

Instead of representing each record using 16 dimensions, I can represent
it using 3 new dimensions called **Principal Components**.

A simple way I understand PCA is:

```text
Many Features
      ↓
     PCA
      ↓
Fewer Dimensions
```

However, reducing dimensions also means that some of the original
variability can be lost.

For this reason, after applying PCA, I need to evaluate how much variability
is represented by the Principal Components.

---

## PCA is Not Feature Selection

One of the most important things I learned is that PCA does **not** simply
select the best original features.

For example, if I have:

```text
f1
f2
f3
f4
...
f16
```

PCA does not do this:

```text
16 Original Features
        ↓
Select the 3 Best Features
        ↓
f2, f7, f12
```

Instead, it creates new variables:

```text
16 Original Features
        ↓
       PCA
        ↓
PC1, PC2, PC3
```

The Principal Components are created mathematically using information from the original features.

So:

```text
Feature Selection
      ↓
Keeps some of the original features


PCA
      ↓
Creates new components from the original features
```

This distinction helped me understand why PCA is considered a
**dimensionality reduction technique** rather than simply a feature
selection technique.

---

## Understanding Principal Components

A **Principal Component** is a new variable created mathematically from
the original features.

The components are usually called:

```text
PC1
PC2
PC3
...
```

Each Principal Component represents a new direction in the data.

Conceptually:

```text
Original Features

f1 ─────┐
f2 ─────┤
f3 ─────┤
f4 ─────┤
...     │
f16 ────┘
         ↓
        PCA
         ↓
    PC1  PC2  PC3
```

The Principal Components are not original columns from the dataset.

For example:

```text
PC1 ≠ f1
PC2 ≠ f2
PC3 ≠ f3
```

Instead, each component combines information from multiple original features.

---

## Understanding Variance

To understand PCA, I also needed to understand the basic idea of **variance**.

In simple terms, variance describes how much values change or spread.

For example:

```text
10
10
10
10
10
```

There is no variation between these values.

Now compare them with:

```text
2
5
10
18
30
```

The values are more spread out, so there is more variability.

PCA looks for directions in the data that capture as much variability as possible.

The Principal Components are ordered according to how much variability they capture:

```text
PC1
 ↓
Captures the largest possible amount of variability

PC2
 ↓
Captures the next largest amount of variability

PC3
 ↓
Captures the next amount

...
```

This is why PC1 normally explains more variance than the following components.

---

## Before PCA

Before reaching this stage, the data was already prepared during Feature Engineering.

That process included:

```text
Select / Create Features
        ↓
Prepare the Features
        ↓
VectorAssembler
        ↓
features_raw
        ↓
StandardScaler
        ↓
features_scaled
```

The details of those transformations are documented in **Machine_Learning_Feature**.

For PCA, the important point is that I already have:

```text
features_scaled
```

This vector becomes the input for PCA.

```text
features_scaled
       ↓
      PCA
```

---

## Loading the Prepared Features

In my implementation, I saved the prepared features as a Delta table during the previous stage.

I can load them using:

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

I can inspect the input vector:

```python
df_features.select(
    "features_scaled"
).show(3, truncate=False)
```

At this point:

```text
features_scaled
       ↓
Ready for PCA
```

---

## Applying PCA with PySpark

PCA is available in PySpark through:

```python
from pyspark.ml.feature import PCA
```

In my implementation, I configured PCA with three Principal Components:

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
| `k` | Number of Principal Components to create |
| `inputCol` | Vector containing the input features |
| `outputCol` | Column where the PCA representation will be stored |

In this case:

```python
k=3
```

means:

```text
Create 3 Principal Components
```

Therefore:

```text
features_scaled

[f1, f2, f3, ... f16]

          ↓
       PCA (k=3)
          ↓

pca_features

[PC1, PC2, PC3]
```

An important thing I learned here is that `k=3` does **not** mean PCA
automatically decided that three components were optimal.

I configured PCA to create three components.

---

## Understanding `fit()` and `transform()`

After configuring PCA, I fit the model:

```python
pca_model = pca.fit(
    df_features
)
```

Then I transform the dataset:

```python
df_pca = pca_model.transform(
    df_features
)
```

I understand these two operations as:

```text
fit()
  ↓
Analyze the data
  ↓
Learn how the Principal Components should be constructed
```

and:

```text
transform()
  ↓
Use the learned transformation
  ↓
Represent each record using the Principal Components
```

After `transform()`, PySpark creates the new column:

```text
pca_features
```

which contains the new PCA representation for each record.

---

## Explained Variance

After applying PCA, the next question is:

> How much of the original variability is represented by each Principal Component?

This is what **explained variance** helps me understand.

In PySpark, I can retrieve it using:

```python
explained_variance = (
    pca_model.explainedVariance
)
```

Then I can display the percentage represented by each component:

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

In my practical implementation, the result was approximately:

```text
PC1 → 20.48%
PC2 → 15.06%
PC3 → 13.00%
```

This does **not** mean:

```text
PC1 contains 20.48%
of the original features
```

It means that:

```text
PC1
 ↓
Represents approximately
20.48% of the variability
in the original feature data
```

The same interpretation applies to PC2 and PC3.

---

## Cumulative Explained Variance

I can also calculate how much variability the components represent together.

For this, I used:

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

In my implementation, I obtained approximately:

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

It does not mean that 48.54% of the features or records were preserved.

It refers specifically to the amount of **variability represented by the three components**.

This also helped me understand an important trade-off:

```text
Fewer Dimensions
       ↓
Simpler Representation

BUT

Fewer Dimensions
       ↓
Some Original Variability
Can Be Lost
```

---

## Choosing the Number of Components

One of the questions I had when learning PCA was:

> How do I know how many components I should use?

There is no universal rule saying that PCA should always use two, three, or any specific number of components.

When I configure:

```python
PCA(k=3)
```

I am explicitly asking PySpark to create three Principal Components.

PCA determines **how those components are constructed**, but I define how many components I want through `k`.

I understand the process like this:

```text
I choose k
    ↓
PCA constructs the components
    ↓
I evaluate the explained variance
    ↓
I decide whether the reduced
representation is useful
```

A practical approach is to evaluate different values of `k`:

```text
Choose k
   ↓
Apply PCA
   ↓
Check Explained Variance
   ↓
Check Cumulative Variance
   ↓
Evaluate the Result
```

The goal is to find an appropriate balance between:

```text
Reduce Dimensionality
        ↕
Preserve Variability
```

In my implementation, I configured:

```text
k = 3
```

and then evaluated the explained variance.

The three components represented approximately 48.54% of the original variability.

This does not mean that three components are the correct choice for every dataset.

---

## PCA with Scikit-learn

PCA is the technique, while PySpark ML is one of the tools that can be used to implement it.

PCA can also be implemented using **Scikit-learn**.

For example:

```python
from sklearn.decomposition import PCA

pca = PCA(
    n_components=3
)

X_pca = pca.fit_transform(
    X_scaled
)
```

In this case:

```text
Scikit-learn
n_components = 3
        ↓
Create 3 Principal Components
```

This is conceptually similar to:

```text
PySpark
k = 3
     ↓
Create 3 Principal Components
```

The syntax is different, but the idea is the same.

With Scikit-learn, the explained variance can be obtained using:

```python
pca.explained_variance_ratio_
```

### Selecting Components by Variance

Scikit-learn also allows another approach.

Instead of defining the exact number of components, I can specify how much variance I want to preserve.

For example:

```python
pca = PCA(
    n_components=0.95
)

X_pca = pca.fit_transform(
    X_scaled
)
```

Conceptually:

```text
n_components = 0.95
        ↓
Preserve approximately
95% of the variance
        ↓
PCA determines how many
components are required
```

This helped me understand that choosing Principal Components can be approached in different ways.

In PySpark ML, `k` represents an integer number of components, so I can evaluate the explained variance for different values of `k` when deciding how many components to keep.

---

## Visualizing the Explained Variance

I can also visualize the explained variance to make the result easier to interpret.

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

plt.xlabel(
    "Principal Components"
)

plt.ylabel(
    "Explained Variance (%)"
)

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

This makes it easier to compare how much variability is represented by
each Principal Component.

---

## Saving the PCA Results

Once PCA is complete, the transformed data can be saved for the next stage of the Machine Learning process.

In my implementation, I saved the PCA result as a Delta table.

A generic example is:

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

Saving the PCA output is not required for PCA itself.

I used this approach because I separated the Machine Learning process into different stages:

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

This made each stage easier to execute and validate separately.

---

## Next Step: K-Means

After PCA, I used the resulting Principal Components as the input for K-Means clustering.

The connection between both stages is simple:

```text
features_scaled
       ↓
PCA
       ↓
pca_features
       ↓
K-Means
```

Instead of using the original feature representation:

```text
[f1, f2, f3, ... f16]
```

K-Means received the reduced PCA representation:

```text
[PC1, PC2, PC3]
```

The next document continues from this point with **K-Means clustering**.

---

## What I Learned

After learning and applying PCA, the main concepts I wanted to keep
documented are:

- PCA is a dimensionality reduction technique.
- PCA does not select the best original features.
- PCA creates new variables called Principal Components.
- Principal Components combine information from the original features.
- PC1 captures the largest possible amount of variability, followed by the
  remaining components.
- In PySpark, `k` defines how many Principal Components I want to create.
- PCA determines how those components are constructed.
- `fit()` learns the PCA transformation.
- `transform()` applies that transformation to the records.
- Explained variance tells me how much variability each component represents.
- Cumulative explained variance tells me how much variability several
  components represent together.
- Reducing dimensionality means that some original variability may be lost.
- There is no universal value of `k` that works for every dataset.
- PCA can be implemented using different libraries, including PySpark ML
  and Scikit-learn.
- The resulting PCA representation can be used as input for other Machine
  Learning algorithms such as K-Means.

The simplest way I remember PCA is:

```text
features_scaled
       ↓
      PCA
       ↓
Principal Components
       ↓
Explained Variance
       ↓
pca_features
       ↓
Next ML Step
```

In my practical implementation:

```text
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

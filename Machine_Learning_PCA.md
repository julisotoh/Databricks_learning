# Principal Component Analysis (PCA) with PySpark

## Based on What I Learned and Applied

This documentation is based on what I learned about Principal Component
Analysis (PCA) and how I applied those concepts using PySpark and Databricks.

My goal is to document PCA in a simple and practical way: what it is, how Principal Components work, how to apply PCA, and how to evaluate the result using explained variance.

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

It is a dimensionality reduction technique used when we have multiple features and want to represent them using fewer dimensions.

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

Instead of representing each record using 16 dimensions, I can represent it using 3 new dimensions called **Principal Components**.

A simple way I understand PCA is:

```text
Many Features
      ↓
     PCA
      ↓
Fewer Dimensions
```

However, reducing dimensions also means that some of the original variability can be lost.

For this reason, after applying PCA, I need to evaluate how much variability is represented by the Principal Components.

---

## PCA is Not Feature Selection

One of the most important things I learned is that PCA does **not** simply select the best original features.

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
Keeps some of the
original features


PCA
      ↓
Creates new components
from the original features
```

This distinction helped me understand why PCA is considered a **dimensionality reduction technique** rather than simply a feature
selection technique.

---

## Understanding Principal Components

A **Principal Component** is a new variable created mathematically from the original features.

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

## How Many Principal Components Should I Use?

When I started working with PCA, one of the first questions I had was:

> If I have 16 features, how do I know how many Principal Components I should keep?

The purpose of PCA is to reduce dimensionality, but reducing the number of dimensions also means deciding how much of the original variability I am willing to leave outside the reduced representation.

For example:

```text
16 Original Features
        ↓
       PCA
        ↓
How many components?

1?
2?
3?
4?
5?
...
```

If I keep many components, I may preserve more variability, but the dimensionality reduction will be smaller.

If I keep very few components, I reduce the dimensionality more, but I may lose more of the original variability.

Conceptually:

```text
More Components
      ↓
More Variability Represented
      ↓
Less Dimensionality Reduction
```

while:

```text
Fewer Components
      ↓
More Dimensionality Reduction
      ↓
Potentially More Variability Lost
```

This helped me understand that there is no universal rule saying that PCA should always use two, three, or any specific number of components.

In PySpark, I define the number of components using `k`.

For example:

```python
PCA(k=3)
```

means that I am asking PCA to create three Principal Components.

An important distinction is:

```text
I choose how many components I want
             ↓
             k

PCA determines how those components should be mathematically constructed
```

So `k=3` does **not** mean that PCA automatically discovered that three components were the optimal number.

A practical way to evaluate the decision is:

```text
Choose k
   ↓
Apply PCA
   ↓
Check Explained Variance
   ↓
Check Cumulative Explained Variance
   ↓
Evaluate the Result
```

In my implementation, I configured:

```text
k = 3
```

I then evaluated the explained variance to understand how much of the original variability was represented by those three components.

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

df_features = spark.table("mart_myproject.ml_features")
```

I can inspect the input vector:

```python
df_features.select("features_scaled").show(3, truncate=False)
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

---

## Why Do We Use `fit()` and `transform()`?

When I first used PCA, I also wanted to understand why the process required
two operations:

```python
fit()
```

and:

```python
transform()
```

The simplest way I understand the difference is:

```text
fit()
  ↓
Learn from the data


transform()
  ↓
Apply what was learned
```

### `fit()`

When I create PCA:

```python
pca = PCA(
    k=3,
    inputCol="features_scaled",
    outputCol="pca_features"
)
```

I have configured how I want PCA to work, but PCA has not yet analyzed the data.

At this point, I have defined:

```text
Use PCA
   ↓
Create 3 components
   ↓
Read features_scaled
   ↓
Create pca_features
```

Then I use:

```python
pca_model = pca.fit(
    df_features
)
```

During `fit()`, PCA analyzes the input data and learns how the Principal Components should be constructed.

Conceptually:

```text
features_scaled
       ↓
      fit()
       ↓
Analyze the variability
in the data
       ↓
Learn the Principal
Component directions
       ↓
pca_model
```

This helped me understand the difference between:

```text
pca
```

and:

```text
pca_model
```

`pca` contains the PCA configuration.

`pca_model` contains the PCA transformation learned from the data.

---

### `transform()`

Once PCA has learned how to construct the Principal Components, I can apply that transformation to the records:

```python
df_pca = pca_model.transform(
    df_features
)
```

`transform()` takes each record and represents it using the Principal Components learned during `fit()`.

Conceptually:

```text
Original Representation

[f1, f2, f3, ... f16]

          ↓
     pca_model
          ↓
     transform()
          ↓

New Representation

[PC1, PC2, PC3]
```

After `transform()`, PySpark creates the new column:

```text
pca_features
```

So I remember the complete process as:

```text
Data
 ↓
fit()
 ↓
Learn the transformation
 ↓
Model
 ↓
transform()
 ↓
Apply the transformation
 ↓
New representation
```

---

## Do We Always Need `fit()` and `transform()`?

Another thing I learned is that not every PySpark ML operation requires both `fit()` and `transform()`.

It depends on whether the operation needs to **learn something from the data**.

For example, PCA needs to analyze the data to learn the Principal Components.

Therefore:

```text
PCA
 ↓
fit()
 ↓
PCA Model
 ↓
transform()
```

`StandardScaler`, which I used during Feature Engineering, follows a similar pattern because it needs to learn statistics from the data before applying the standardization.

```text
StandardScaler
      ↓
     fit()
      ↓
Scaler Model
      ↓
 transform()
```

However, `VectorAssembler` does not need to learn parameters from the data.

Its job is to combine selected columns into a vector.

Therefore, I can use:

```python
df_vectorized = vector_assembler.transform(
    df_features
)
```

without calling `fit()` first.

This introduced me to two important concepts in PySpark ML:

### Estimator

An **Estimator** needs to learn something from the data.

Conceptually:

```text
Estimator
    ↓
   fit()
    ↓
Model / Transformer
    ↓
transform()
```

PCA and StandardScaler are examples of this pattern.

### Transformer

A **Transformer** already knows how to apply its transformation and can use `transform()` directly.

Conceptually:

```text
Transformer
     ↓
transform()
     ↓
Transformed Data
```

`VectorAssembler` is an example.

A simple way I remember the difference is:

```text
Does it need to learn something
from the data?

        YES
         ↓
        fit()
         ↓
       Model
         ↓
    transform()


         NO
         ↓
    transform()
```

This helped me understand that `fit()` and `transform()` are not just PCA commands. They are part of the way many components of the PySpark ML API are designed.

---

## Explained Variance

After applying PCA, the next question is:

> How much of the original variability is represented by each Principal Component?

This is what **explained variance** helps me understand.

In PySpark, I can retrieve it using:

```python
explained_variance = (pca_model.explainedVariance)
```

Then I can display the percentage represented by each component:

```python
for i, variance in enumerate(explained_variance, start=1):
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
PC1 contains 20.48% of the original features
```

It means:

```text
PC1
 ↓
Represents approximately 20.48% of the variability in the original feature data
```

The same interpretation applies to PC2 and PC3.

---

## Cumulative Explained Variance

Looking at each component separately is useful, but I also want to know
how much variability the components represent together.

For this, I used:

```python
cumulative_variance = np.cumsum(explained_variance)
```

Then:

```python
for i, variance in enumerate(cumulative_variance,start=1):
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
48.54% Cumulative Explained Variance
```

This means that the three Principal Components represented approximately **48.54% of the variability contained in the original feature set**.

It does not mean that 48.54% of the features or records were preserved.

It refers specifically to the amount of **variability represented by the three components**.

This helped me understand the trade-off I was considering when choosing the number of components:

```text
Fewer Dimensions
       ↓
Simpler Representation

BUT

Fewer Dimensions
       ↓
Some Original Variability Can Be Lost
```

In my implementation, I used three components and then evaluated the result.

The 48.54% cumulative explained variance describes how much variability those three components represented. It does not demonstrate that `k=3` was the optimal number of components.

---

## PCA with Scikit-learn

PCA is the technique, while PySpark ML is one of the tools that can be used to implement it.

PCA can also be implemented using **Scikit-learn**.

For example:

```python
from sklearn.decomposition import PCA

pca = PCA( n_components=3)

X_pca = pca.fit_transform( X_scaled)
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
pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)
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

I can also visualize the explained variance to make the PCA result easier to interpret.

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

This makes it easier to compare how much variability is represented by each Principal Component.

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

The connection between both stages is:

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



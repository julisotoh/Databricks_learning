# Principal Component Analysis (PCA) with PySpark

## Based on What I Learned and Applied

This documentation is based on what I learned about Principal Component
Analysis (PCA) during my Master's in Big Data & Artificial Intelligence
and how I later applied those concepts using PySpark and Databricks.

My goal is to document PCA in a simple and practical way: what it is,
why it can be useful, how to prepare the data, how to apply it, and how
to understand the results.

The process documented here follows the same approach I used in practice:

```text
Prepared Data
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
      ↓
Principal Components
      ↓
Explained Variance
      ↓
pca_features
      ↓
K-Means
```

The examples are simplified to make the concepts easier to understand,
while some results are included from my practical implementation in
Databricks.

---

## What is PCA?

PCA stands for **Principal Component Analysis**.

It is a dimensionality reduction technique used when we have multiple
features and want to represent them using fewer dimensions.

For example, imagine that after Feature Engineering I have:

```text
16 features
```

Each record is represented using those 16 characteristics:

```text
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

Instead of working with 16 dimensions, I can now represent each record
using 3 new dimensions.

A simple way I remember PCA is:

```text
Many Features
      ↓
     PCA
      ↓
Smaller Representation
of the Data
```

However, reducing dimensions also means that I need to evaluate how much
of the original variability is preserved.

---

## Understanding PCA with a Simple Example

Imagine that I have a dataset from a sushi restaurant.

After preparing the data, I could have features such as:

```text
sushi_orders
delivery_orders
average_price
weekend_orders
night_orders
repeat_orders
total_sales
customer_frequency
...
```

Each feature describes something different about the behavior of the
customers or their orders.

If I continue creating useful features, I could eventually have many
dimensions.

For example:

```text
sushi_orders ─────────┐
delivery_orders ──────┤
average_price ────────┤
weekend_orders ───────┤
night_orders ─────────┤
repeat_orders ────────┤
total_sales ──────────┤
customer_frequency ───┤
...                    │
                       ↓
                      PCA
                       ↓
               ┌───────┼───────┐
               ↓       ↓       ↓
              PC1     PC2     PC3
```

PCA creates a smaller representation of the original data.

But there is something very important:

**PCA does not select the three best original features.**

For example:

```text
PC1 ≠ sushi_orders
PC2 ≠ total_sales
PC3 ≠ delivery_orders
```

Instead, PCA creates new variables using information from the original
features.

This helped me understand the difference between PCA and Feature Selection:

```text
Feature Selection
       ↓
Select some of the
original features


PCA
       ↓
Create new components
using information from
the original features
```

---

## What is a Principal Component?

A **Principal Component** is a new variable created mathematically from the
original features.

The components are called:

```text
PC1
PC2
PC3
...
```

Each component represents a direction in the data that captures part of its
variability.

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

This means that the Principal Components are not original columns from the
dataset.

They are a new representation of the information contained in those
features.

---

## Understanding Variance

To understand PCA, I also needed to understand the basic idea of
**variance**.

In simple terms, variance describes how much the values in the data change
or spread.

For example:

```text
10
10
10
10
10
```

These values do not change.

Now compare them with:

```text
2
5
10
18
30
```

These values have more variation.

PCA looks for directions in the data where there is more variability.

The Principal Components are ordered according to how much variability
they capture:

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

This is why the first Principal Component usually explains more variance
than the following components.

---

## Preparing the Data Before PCA

Before applying PCA, the data should already be prepared.

In the process I followed, Feature Engineering happened before PCA:

```text
Gold / MART Data
       ↓
Prepare Dataset
       ↓
Select / Create Features
       ↓
Handle Missing Values
       ↓
Transform Categorical Variables
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

The PCA input was therefore not the original DataFrame.

It was the vector:

```text
features_scaled
```

created during Feature Engineering.

---

## Why Use StandardScaler Before PCA?

This was an important part of the process because PCA is affected by the
scale of the variables.

Imagine that I have these features:

```text
binary_feature        → 0 / 1
average_speed         → 35
traffic_intensity     → 1200
```

The numerical ranges are very different.

I do not want a variable to have more influence on the analysis simply
because its numerical values are much larger.

For this reason, I first used:

```text
VectorAssembler
       ↓
features_raw
       ↓
StandardScaler
       ↓
features_scaled
```

and then used `features_scaled` as the input for PCA.

Conceptually:

```text
Different Numerical Scales
          ↓
    StandardScaler
          ↓
Standardized Features
          ↓
         PCA
```

---

## Loading the Prepared Features

Once the features are prepared, I can load the dataset that contains the
standardized feature vector.

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

I can inspect the vector using:

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

I can configure it using:

```python
pca = PCA(
    k=3,
    inputCol="features_scaled",
    outputCol="pca_features"
)
```

There are three important parameters:

```text
k
↓
Number of Principal Components I want to create


inputCol
↓
Vector containing the prepared features


outputCol
↓
Column where the new PCA
representation will be stored
```

In this example:

```python
k=3
```

means:

> Create three Principal Components.

Conceptually:

```text
Original Representation

[f1, f2, f3, ... f16]

          ↓
       PCA (k=3)
          ↓

New Representation

[PC1, PC2, PC3]
```

An important thing I learned is that:

```text
k = 3
```

does **not** mean that PCA automatically decided that three components were
the optimal number.

I configured PCA to create three components.

---

## Understanding `fit()` and `transform()`

After configuring PCA, I first fit the model:

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
Analyze the data and learn how the Principal Components should be constructed


transform()
  ↓
Use those components to represent each record in the new PCA space
```

After `transform()`, PySpark creates:

```text
pca_features
```

So the transformation looks like:

```text
features_scaled

[f1, f2, f3, ... f16]

          ↓
         PCA

pca_features

[PC1, PC2, PC3]
```

---

## Explained Variance

After applying PCA, it is important to understand how much of the original
variability is represented by the Principal Components.

In PySpark, I can check this using:

```python
explained_variance = (
    pca_model.explainedVariance
)
```

Then I can display the result:

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

In the implementation I developed in Databricks, I worked with three
Principal Components and obtained approximately:

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

It means:

```text
PC1
 ↓
Represents approximately
20.48% of the variability
in the original feature data
```

---

## Cumulative Explained Variance

Looking at each component separately is useful, but I also want to know how
much variability the components represent together.

For this, I can calculate the cumulative explained variance:

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

In my practical implementation, the result was approximately:

```text
PC1
20.48%

PC1 + PC2
35.54%

PC1 + PC2 + PC3
48.54%
```

So the process was:

```text
16 Original Features
        ↓
StandardScaler
        ↓
PCA (k=3)
        ↓
3 Principal Components
        ↓
48.54%
Cumulative Explained Variance
```

This helped me understand an important part of PCA:

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

In this case, the three components represented approximately **48.54% of
the variability contained in the original feature set**.

---

## Choosing the Number of Components

One of the questions I had when learning PCA was:

> How do I know how many components I should use?

There is no universal rule saying:

```text
PCA must always use 2 components
```

or:

```text
PCA must always use 3 components
```

When I configure:

```python
PCA(k=3)
```

I am explicitly asking PySpark to create three Principal Components.

A practical way to evaluate the number of components is:

```text
Choose a value of k
        ↓
Apply PCA
        ↓
Check Explained Variance
        ↓
Check Cumulative Variance
        ↓
Evaluate how much variability
is being preserved
```

Depending on the dataset and the objective, different values of `k` can be
evaluated.

The idea is to find a useful balance between:

```text
Reduce Dimensionality
        ↕
Preserve Variability
```

In my Databricks implementation, I configured:

```text
k = 3
```

and then evaluated the explained variance.

This does not mean that three components would necessarily be the correct
choice for another dataset.

---

## PCA Can Also Be Applied with Scikit-learn

PCA is a Machine Learning technique, so the concept is not exclusive to
PySpark.

When working with a Python Machine Learning workflow, PCA can also be
implemented using **Scikit-learn**.

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

Here:

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
```

The syntax is different, but both are asking PCA to create three Principal
Components.

With Scikit-learn, I can check the explained variance using:

```python
pca.explained_variance_ratio_
```

Scikit-learn also allows another approach.

Instead of specifying an exact number of components, I can specify how much
variance I want to preserve.

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

This helped me understand that there are different ways to think about the
number of Principal Components.

In PySpark ML, `k` represents an integer number of components, so I can
evaluate the explained variance and compare different values of `k` when
deciding how many components to keep.

---

## Visualizing the Explained Variance

I can also visualize the explained variance to make the PCA result easier
to interpret.

```python
variance_pct = (
    explained_variance * 100
)

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

This makes it easier to compare how much variability is represented by each
Principal Component.

---

## Saving the PCA Results

Once PCA is complete, the transformed data can be saved for the next stage
of the Machine Learning process.

In Databricks, I saved the PCA result as a Delta table.

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

In my case, I used this approach because I separated the Machine Learning
process into different stages:

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

This made each stage easier to execute, validate, and understand.

---

## Using PCA with K-Means

After PCA, I used the resulting components as the input for K-Means
clustering.

Instead of using the original feature vector:

```text
[f1, f2, f3, ... f16]
```

K-Means received:

```text
[PC1, PC2, PC3]
```

So the complete Machine Learning flow became:

```text
16 Features
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
      ↓
3 Principal Components
      ↓
pca_features
      ↓
K-Means
```

This allowed me to use the reduced PCA representation as the input for the
clustering process.

---

## Complete Flow

The complete process can be summarized as:

```text
Gold / MART Data
       ↓
Feature Engineering
       ↓
Select / Create Features
       ↓
Handle Missing Values
       ↓
Transform Categorical Variables
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
       ↓
Principal Components
       ↓
Explained Variance
       ↓
Cumulative Explained Variance
       ↓
pca_features
       ↓
K-Means
```

---

## What I Learned

After learning and applying PCA, these are the concepts that helped me
understand it better:

- PCA is a dimensionality reduction technique.
- PCA does not simply select the best original features.
- PCA creates new variables called Principal Components.
- Principal Components combine information from the original features.
- PC1 captures the largest possible amount of variability, followed by PC2,
  PC3, and the remaining components.
- Features should usually be standardized when their numerical scales are
  very different.
- In PySpark, `k` defines how many Principal Components will be created.
- `fit()` learns how the Principal Components should be constructed.
- `transform()` represents each record using those components.
- Explained variance tells me how much variability each component represents.
- Cumulative explained variance tells me how much variability the components
  represent together.
- Reducing dimensionality creates a simpler representation, but some
  variability can be lost.
- There is no universal number of Principal Components that works for every
  dataset.
- PCA can be implemented using different Machine Learning libraries, such as
  PySpark ML and Scikit-learn.
- The PCA output can be used as input for other Machine Learning algorithms,
  such as K-Means.

The simplest way I remember the complete idea is:

```text
Many Features
      ↓
Standardize
      ↓
PCA
      ↓
Fewer Principal Components
      ↓
Check Explained Variance
      ↓
Use the Reduced Representation
in the Next ML Step
```

In my practical implementation:

```text
16 Features
      ↓
StandardScaler
      ↓
PCA (k=3)
      ↓
3 Principal Components
      ↓
48.54% Cumulative Explained Variance
      ↓
K-Means
```

This is how I connected the concepts I learned about PCA with a practical
Machine Learning implementation using PySpark and Databricks.

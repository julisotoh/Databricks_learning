# Principal Component Analysis (PCA) with PySpark

## Based on What I Learned and Applied

This documentation continues from the **Feature Engineering** process.

During Feature Engineering, I learned that the features I use depend on
the objective of the Machine Learning problem.

Now the question is:

> Once I have my features prepared, why would I use PCA?

Understanding this question helped me understand the purpose of
Principal Component Analysis.

The process I followed was:

```text
Feature Engineering
        ↓
VectorAssembler
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

## What is PCA?

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

PCA allows me to create a smaller representation of my original feature
space.

However, PCA is **not mandatory** in every Machine Learning process.

Having many features does not automatically mean that I need PCA.

---

# Understanding PCA with a Sushi Restaurant Example

I can continue with the same type of example I used while learning
Feature Engineering.

Imagine that a sushi restaurant wants to understand the behavior of its
customers and later identify groups of customers with similar
characteristics.

After Feature Engineering, I could have features such as:

```text
Customer Age
Orders per Month
Average Order Price
Monthly Spending
Sushi Orders
Other Food Orders
Uses Promotions
Orders on Weekends
```

Each feature gives me information about the customer.

Conceptually:

```text
Customer
   ↓
Age
Orders per Month
Average Order Price
Monthly Spending
Sushi Orders
Other Food Orders
Uses Promotions
Orders on Weekends
```

In this example I have:

```text
8 Features
    ↓
8 Dimensions
```

The number of features represents the number of dimensions that I am
using to describe each customer.

---

## First: Why StandardScaler?

Before PCA, the features have already been prepared during Feature
Engineering.

However, they can have very different numerical scales.

For example:

```text
Age                  → 35
Orders per Month     → 12
Average Order Price  → 37,500
Monthly Spending     → 450,000
Sushi Orders         → 8
Uses Promotions      → 1
```

`Monthly Spending` has a much larger numerical magnitude than
`Uses Promotions`, `Orders per Month`, or `Age`.

For algorithms that are sensitive to scale, I do not want one feature to
have more influence simply because its numerical values are much larger.

For this reason, I first standardize the features.

```text
Features
    ↓
VectorAssembler
    ↓
features_raw
    ↓
StandardScaler
    ↓
features_scaled
```

After this step, the features are on comparable scales.

The resulting:

```text
features_scaled
```

becomes the input for PCA.

---

# Why Would I Use PCA?

Now suppose I have the eight prepared features from the sushi example.

I could continue directly to K-Means:

```text
8 Features
    ↓
StandardScaler
    ↓
K-Means
```

This is completely possible.

**PCA is not required before K-Means.**

But I can also ask:

> Can I represent the information contained in these features using fewer
> dimensions?

For example:

```text
8 Original Features
        ↓
       PCA
        ↓
PC1   PC2   PC3
```

Instead of representing each customer using eight dimensions, I now
represent each customer using three new dimensions.

I think of PCA as creating a **smaller representation of the original
feature space**.

Or, in a simpler way:

```text
8 Features
    ↓
   PCA
    ↓
3 Components
```

This can be useful when I have many features and I want to reduce the
dimensionality before applying another Machine Learning algorithm.

---

# PCA Does Not Select the Best Features

This was one of the most important things for me to understand.

At first, it is easy to think that PCA does something like this:

```text
8 Original Features
        ↓
       PCA
        ↓
Select the 3 Best Features
        ↓
Age
Monthly Spending
Sushi Orders
```

But that is **not** what PCA does.

PCA creates completely new variables called **Principal Components**.

```text
8 Original Features
        ↓
       PCA
        ↓
PC1
PC2
PC3
```

Therefore:

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

For example:

```text
PC1 ≠ Age

PC2 ≠ Monthly Spending

PC3 ≠ Sushi Orders
```

Each Principal Component is mathematically created using information
from the original features.

That is why PCA is a **dimensionality reduction technique**, not simply
a feature selection technique.

---

# PCA Does Not Create Groups

Another important distinction for me was understanding that the
Principal Components are **not clusters**.

After PCA:

```text
Customer A → [PC1, PC2, PC3]

Customer B → [PC1, PC2, PC3]

Customer C → [PC1, PC2, PC3]
```

I have changed the way each customer is represented.

I have **not grouped the customers yet**.

The grouping happens later with K-Means.

I remember the difference like this:

```text
PCA
 ↓
Can I represent my features
with fewer dimensions?


K-Means
 ↓
Which records are similar
to each other?
```

---

# The Trade-Off When Using PCA

Reducing dimensions sounds useful, but there is a trade-off.

If I keep more Principal Components:

```text
More Components
      ↓
More Original Variability Represented
      ↓
Less Dimensionality Reduction
```

If I keep fewer Principal Components:

```text
Fewer Components
      ↓
More Dimensionality Reduction
      ↓
Potentially More Variability Lost
```

Therefore, PCA is not simply:

```text
More reduction = better
```

I need to evaluate how much of the original variability is represented
after reducing the dimensions.

This is where **explained variance** becomes important.

---

# How Many Principal Components Should I Use?

Once I understood why PCA reduces dimensions, my next question was:

> If I have several features, how many Principal Components should I keep?

Suppose I have:

```text
8 Original Features
        ↓
       PCA
        ↓
How many components?

2?
3?
4?
5?
```

There is no universal number that I must always use.

In PySpark, I define the number of Principal Components using `k`.

For example:

```python
PCA(k=3)
```

means that I want PCA to create:

```text
PC1
PC2
PC3
```

An important distinction is:

```text
I choose k
    ↓
PCA learns how the components
should be constructed
```

Therefore:

```text
k = 3
```

does **not** mean that PCA automatically discovered that three components
were the optimal number.

I still need to evaluate the result.

---

# Understanding Explained Variance

Suppose I apply PCA to the sushi customer example and obtain:

```text
PC1 → 35%
PC2 → 22%
PC3 → 15%
```

Each percentage tells me how much of the original variability is
represented by that Principal Component.

For example:

```text
PC1 → 35%
```

does **not** mean:

```text
PC1 contains 35% of the features
```

It means that PC1 represents approximately 35% of the variability
contained in the original feature data.

---

## Cumulative Explained Variance

Now I can add the variance represented by the components:

```text
PC1 → 35%
PC2 → 22%
PC3 → 15%

35% + 22% + 15% = 72%
```

Therefore:

```text
8 Original Features
        ↓
       PCA
        ↓
3 Principal Components
        ↓
72% Cumulative Explained Variance
```

This means:

> The three Principal Components represent approximately 72% of the
> variability contained in the original features.

It does **not** mean:

```text
PCA is 72% accurate
```

It also does **not** mean:

```text
72% of the features were preserved
```

The 72% refers specifically to the amount of **variability represented
by the three Principal Components**.

So I can think about the result as:

```text
8 Dimensions
      ↓
     PCA
      ↓
3 Dimensions
      ↓
72% of the original variability
is represented
```

The remaining variability is not represented by those three components.

This helps me evaluate the trade-off between reducing dimensions and
preserving variability.

---

# Applying PCA with PySpark and Databricks

In my implementation, I start from the features prepared during Feature
Engineering.

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

I can inspect the vector that will be used as input:

```python
df_features.select(
    "features_scaled"
).show(
    3,
    truncate=False
)
```

At this point:

```text
features_scaled
       ↓
Ready for PCA
```

---

## Configuring PCA

PCA is available in PySpark through:

```python
from pyspark.ml.feature import PCA
```

I configure the PCA transformation:

```python
pca = PCA(
    k=3,
    inputCol="features_scaled",
    outputCol="pca_features"
)
```

The parameters are:

| Parameter | Purpose |
|---|---|
| `k` | Number of Principal Components |
| `inputCol` | Vector containing the input features |
| `outputCol` | Column containing the PCA representation |

Therefore:

```text
features_scaled

[f1, f2, f3, ... f16]

          ↓
       PCA(k=3)
          ↓

pca_features

[PC1, PC2, PC3]
```

---

# Why Do I Need `fit()` and `transform()`?

When I first used PCA, I also wanted to understand why I needed two
operations:

```python
fit()
```

and:

```python
transform()
```

The simplest way I remember them is:

```text
fit()
  ↓
Learn from the data


transform()
  ↓
Apply what was learned
```

---

## `fit()`

When I create:

```python
pca = PCA(
    k=3,
    inputCol="features_scaled",
    outputCol="pca_features"
)
```

I have only configured PCA.

PCA has not learned anything from my data yet.

Then I use:

```python
pca_model = pca.fit(
    df_features
)
```

During `fit()`, PCA analyzes the data and learns how the Principal
Components should be constructed.

Conceptually:

```text
features_scaled
       ↓
      fit()
       ↓
Learn the Principal
Component directions
       ↓
pca_model
```

So:

```text
pca
```

is my PCA configuration.

While:

```text
pca_model
```

contains what PCA learned from the data.

---

## `transform()`

Once PCA has learned the transformation, I can apply it to the records:

```python
df_pca = pca_model.transform(
    df_features
)
```

`transform()` represents each record using the Principal Components
learned during `fit()`.

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

After `transform()`, PySpark creates:

```text
pca_features
```

So I remember the process as:

```text
Data
 ↓
fit()
 ↓
Learn
 ↓
Model
 ↓
transform()
 ↓
Apply
 ↓
New Representation
```

---

# Do I Always Need `fit()` and `transform()`?

No.

It depends on whether the operation needs to learn something from the
data.

For example:

```text
PCA
 ↓
fit()
 ↓
PCA Model
 ↓
transform()
```

`StandardScaler` follows a similar pattern because it needs to learn
statistics from the data before applying the standardization.

```text
StandardScaler
      ↓
     fit()
      ↓
Scaler Model
      ↓
transform()
```

However, `VectorAssembler` does not need to learn parameters from the
data.

Its job is to combine selected columns into one feature vector.

Therefore, I can use:

```python
df_vectorized = vector_assembler.transform(
    df_features
)
```

without calling `fit()` first.

A simple way I remember this is:

```text
Does it need to learn
something from the data?

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

---

# Getting Explained Variance in PySpark

After fitting PCA, I can retrieve the explained variance:

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

---

# My Practical Result

In my practical implementation, I used three Principal Components.

The result was approximately:

```text
PC1 → 20.48%
PC2 → 15.06%
PC3 → 13.00%
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

This means that the three Principal Components represented approximately
**48.54% of the variability contained in the original feature set**.

It does not mean that:

```text
PCA was 48.54% accurate
```

and it does not mean that:

```text
48.54% of the features were preserved
```

It specifically describes how much of the original variability was
represented by those three components.

Also, this result does not demonstrate that:

```text
k = 3
```

was automatically the optimal number of components.

It describes the result of the dimensionality reduction that I chose to
evaluate.

---

# Visualizing Explained Variance

I can also visualize the explained variance to make the PCA result easier
to interpret.

```python
variance_pct = explained_variance * 100

components = [
    f"PC{i}"
    for i in range(
        1,
        len(variance_pct) + 1
    )
]

plt.figure(
    figsize=(9, 6)
)

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
        bar.get_x()
        + bar.get_width() / 2,
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

# PCA with Python / Scikit-learn

PCA is the technique.

PySpark is only one of the tools that I can use to implement it.

I can also apply PCA using Python with Scikit-learn.

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

The syntax changes, but the idea is the same.

I can obtain the explained variance using:

```python
pca.explained_variance_ratio_
```

---

## Selecting Components by Variance with Scikit-learn

Scikit-learn also allows me to approach the problem differently.

Instead of specifying exactly how many components I want, I can specify
how much variance I want to preserve.

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

This is another way of answering the question:

> How many Principal Components should I keep?

In PySpark ML, `k` represents an integer number of components, so I can
evaluate the explained variance obtained with different values of `k`.

---

# Saving the PCA Results in Databricks

Once PCA is complete, I can save the transformed data for the next stage
of the Machine Learning process.

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

Saving the PCA result is not required by PCA itself.

I used this approach because I separated the Machine Learning process
into stages:

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

This allows me to execute and validate each stage separately.

---

# Connecting PCA with K-Means

This was the connection that helped me understand the purpose of PCA in
my Machine Learning process.

Going back to the sushi example, originally each customer could be
represented using:

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

After PCA, the same customer could be represented as:

```text
Customer
   ↓
[PC1, PC2, PC3]
```

PCA has **not grouped the customers**.

It has only created a smaller representation of their original features.

Then K-Means receives:

```text
Customer A → [PC1, PC2, PC3]

Customer B → [PC1, PC2, PC3]

Customer C → [PC1, PC2, PC3]
```

and uses those values to find records that are similar to each other.

Therefore:

```text
PCA
 ↓
Reduce the dimensionality


K-Means
 ↓
Find groups of similar records
```

I could also apply K-Means without PCA:

```text
Feature Engineering
        ↓
StandardScaler
        ↓
K-Means
```

But in my implementation I used:

```text
Feature Engineering
        ↓
StandardScaler
        ↓
PCA
        ↓
K-Means
```

So K-Means received the PCA representation:

```text
[PC1, PC2, PC3]
```

instead of the original feature vector.

---

# What I Learned

The main ideas I learned from PCA are:

- PCA is a dimensionality reduction technique.
- The number of features represents the dimensions of my feature space.
- Having many features does not automatically mean that I need PCA.
- PCA is not mandatory before K-Means.
- PCA does not select the best original features.
- PCA creates new variables called Principal Components.
- Principal Components are not clusters.
- I choose the number of components using `k` in PySpark.
- PCA determines how those components are mathematically constructed.
- `fit()` learns the PCA transformation from the data.
- `transform()` applies what was learned to the records.
- Explained variance tells me how much variability each component
  represents.
- Cumulative explained variance tells me how much variability the
  components represent together.
- Reducing dimensions gives me a smaller representation, but some of the
  original variability may be lost.
- PCA creates the reduced representation.
- K-Means uses the representation to find groups of similar records.

The complete process I applied can be summarized as:

```text
Feature Engineering
        ↓
16 Features
        ↓
VectorAssembler
        ↓
StandardScaler
        ↓
features_scaled
        ↓
PCA (k=3)
        ↓
PC1 + PC2 + PC3
        ↓
48.54% Cumulative Explained Variance
        ↓
K-Means
        ↓
Clusters
```

And the simplest way I now understand the three main stages is:

```text
FEATURE ENGINEERING

What information do I want
to give the model?

        ↓

STANDARD SCALER

Are my numerical features
on comparable scales?

        ↓

PCA

Can I represent this information
with fewer dimensions?

        ↓

K-MEANS

Which records have
similar behavior?
```

# Machine Learning — My Learning Summary

This is a short summary of the concepts I learned and how I currently understand them.

The objective is not to explain all the implementation or code again, but to keep the main concepts clear and connected.

---

## Medallion Architecture

I use Medallion Architecture to separate the data process into layers instead of doing everything in one place.

This makes the solution easier to maintain, trace and troubleshoot.

- **Bronze:** keeps the data as close as possible to the source. If something is wrong later, I can return to Bronze and verify what originally arrived.
- **Silver:** cleans and transforms the data. Here I can handle nulls, standardize values, correct data types, add audit information and apply other transformations.
- **Gold:** contains information prepared for business consumption, reporting, analytics or Machine Learning.

My main idea is:

```text
Bronze → What did I receive?
Silver → How do I clean and prepare it?
Gold   → How will the information be consumed?
```

---

## Feature Engineering

Before selecting features, I need to understand the objective of the analysis.

I should not use every available column just because it exists.

I select the variables that contain useful information for what I want the model to learn or discover.

An ID can be useful to identify a record and join results later, but it normally should not be used as a behavioral feature.

If I have categorical variables that are relevant to the objective, I need to convert them into a numerical representation before using them in algorithms that require numerical input.

---

## VectorAssembler

`VectorAssembler` combines the selected features into a single vector column.

```text
feature_1
feature_2
feature_3
feature_4
      ↓
VectorAssembler
      ↓
features_raw
```

It does not train the model and it does not standardize the values.

It prepares the features in the vector format expected by PySpark ML algorithms.

---

## StandardScaler

`StandardScaler` standardizes the features so that variables with very different numerical scales are more comparable.

This is especially important for algorithms based on distance, such as K-Means, because a variable with much larger numerical values could otherwise have too much influence.

With `withMean=True` and `withStd=True`, the standardization is based on the mean and standard deviation.

It does not mean that the values become equal or that they are converted to a 0–1 range.

```text
VectorAssembler
      ↓
features_raw
      ↓
StandardScaler
      ↓
features_scaled
```

---

## PCA

PCA is used to reduce dimensionality.

It does **not** select the best original features.

Instead, PCA creates new components that represent part of the variability contained in the original features.

```text
Many original features
        ↓
PCA
        ↓
Fewer principal components
```

I decide the number of components using `k`.

For example:

```text
PCA(k=4)
```

means that I want PCA to create four principal components.

### Explained Variance

Explained variance tells me how much of the variability in the original features is represented by the principal components.

For example:

```text
Cumulative explained variance = 70%
```

means that the selected principal components represent approximately 70% of the variability contained in the original features.

It does **not** mean:

```text
70% accuracy
70% of the records
70% of the features
```

If I reduce 25 features to 4 components:

```text
25 → 4
```

that is the dimensionality reduction.

If those four components explain 70% of the variance, approximately 30% of the original variability is not represented by those components.

PCA is **not mandatory before K-Means**.

Whether I use it depends on the number and characteristics of my features and whether reducing dimensions is useful for my analysis.

---

## fit() and transform()

The easiest way for me to remember them is:

```text
fit()
→ learns from the data

transform()
→ applies what was learned
```

For example, PCA uses `fit()` to learn the transformation from the data and `transform()` to apply that transformation.

If the model has already learned and new records arrive, I use the learned model to transform those new records rather than learning everything again.

---

## K-Means

K-Means is an unsupervised Machine Learning algorithm.

It is unsupervised because I do not have a known target telling the model which group each record belongs to.

I use K-Means when I want to discover groups of records with similar characteristics or behavior.

```text
No known target
      ↓
Discover similar groups
      ↓
K-Means
```

### K

In K-Means:

```text
K = number of clusters
```

This is different from PCA:

```text
PCA k
→ number of principal components

K-Means k
→ number of clusters
```

The number of features does not determine the number of clusters.

I decide which values of K I want to test.

---

## Centroids and Distance

Each cluster has a centroid, which represents its center.

K-Means compares the distance between a record and the centroids.

The record is assigned to the cluster whose centroid has the **smallest distance**.

My rule to remember:

```text
SMALLEST DISTANCE
        ↓
CLOSEST CENTROID
        ↓
CLUSTER ASSIGNMENT
```

The cluster number itself does not have an order or meaning.

Cluster 0 is not better or worse than Cluster 1.

---

## Silhouette

Silhouette helps me evaluate how well defined the groups created by K-Means are.

It considers two things:

```text
COHESION
How well the records fit inside their own cluster

+

SEPARATION
How separated the clusters are from each other
```

When I test different values of K, I can compare their Silhouette Scores.

In general, among the K values I tested, a higher Silhouette indicates a better-defined clustering solution.

Important:

```text
Silhouette is NOT accuracy.
```

A Silhouette of `0.81` does not mean 81% accuracy.

It means that the clustering solution has relatively strong cohesion within the clusters and separation between the clusters.

---

## The Two Rules I Need to Remember

These were easy to confuse at first:

```text
Assign a record to a cluster
→ choose the SMALLEST distance to a centroid
```

but:

```text
Compare different K values
→ generally look for the HIGHEST Silhouette among the values tested
```

So:

```text
Distance   → smaller
Silhouette → higher
```

---

## Interpreting the Clusters

K-Means gives me a cluster assignment such as:

```text
prediction = 0
prediction = 1
prediction = 2
```

But K-Means does not tell me what those groups mean.

The cluster numbers are only identifiers.

To understand each cluster, I return to the original variables and analyze the records that belong to each group.

I can:

```text
JOIN the prediction with the original data
        ↓
GROUP BY cluster
        ↓
Calculate counts, averages or distributions
        ↓
Understand the characteristics of each group
```

Then I can describe what each cluster represents from a business perspective.

If I used PCA for the clustering, I can still return to the original variables to interpret the clusters because they are easier to understand than PC1, PC2, PC3, etc.

K-Means discovers the groups.

I analyze what those groups mean.

The business decides what actions to take from that information.

---

# How I Connect Everything

```text
Understand the objective
        ↓
Select the features
        ↓
Encode categorical variables if necessary
        ↓
VectorAssembler
        ↓
StandardScaler
        ↓
PCA
(optional)
        ↓
K-Means
        ↓
Test different K values
        ↓
Silhouette
        ↓
Select a K among the tested values
        ↓
Cluster prediction
        ↓
Return to original variables
        ↓
Interpret the clusters
```

---

# What I Need to Remember

```text
Feature Engineering
→ Which information is relevant to my objective?

VectorAssembler
→ Combine the features into one vector.

StandardScaler
→ Make numerical scales comparable.

PCA
→ Reduce dimensions.

Explained Variance
→ How much original variability is represented?

K-Means
→ Discover similar groups.

K
→ Number of clusters.

Centroid
→ Center of a cluster.

Distance
→ The smallest distance determines the cluster assignment.

Silhouette
→ How well defined are the groups?

Interpretation
→ Return to the original variables to understand each cluster.
```

---


## Supervised Machine Learning

```text
Unsupervised
→ I do not know the groups
→ I want to discover them

Supervised
→ I have a known target
→ I want to predict it
```


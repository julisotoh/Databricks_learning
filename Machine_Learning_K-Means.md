# K-Means Clustering with PySpark

## Based on What I Learned and Applied

This documentation continues from the **Feature Engineering** and **PCA** stages.

In my Machine Learning workflow, I first prepared and standardized the features, then applied PCA, and finally used the PCA representation as the input for K-Means clustering.

To make K-Means easier to understand, I continue using the **sushi restaurant example** and then connect the concepts with the **parking-demand implementation from my thesis**.

> **Learning note**
>
> This repository documents what I implemented and learned while working with Databricks, PySpark, and Spark ML.
>
> The sushi examples are simplified learning examples. The thesis sections show how I applied the concepts in my academic project.

The complete flow is:

```text
Feature Engineering
        ↓
StandardScaler
        ↓
PCA
        ↓
pca_features
        ↓
K-Means
        ↓
Silhouette
        ↓
Interpret the Clusters
```

---

# What I Learned

## 1. Before K-Means: Do I Always Need PCA?

No. **PCA is not mandatory before K-Means.**

This was an important concept for me because K-Means can be applied directly to standardized features:

```text
Feature Engineering
        ↓
StandardScaler
        ↓
K-Means
```

or after dimensionality reduction:

```text
Feature Engineering
        ↓
StandardScaler
        ↓
PCA
        ↓
K-Means
```

The easiest way for me to remember the difference is:

```text
PCA
 ↓
Can I represent this information
with fewer dimensions?


K-Means
 ↓
Which records have
similar behavior?
```

In my thesis workflow, I used the PCA representation as the input for K-Means.

---

# 2. What Is K-Means?

K-Means is an **unsupervised Machine Learning algorithm** that creates groups of similar records.

It is unsupervised because there is no known target telling the algorithm which group is correct for each record.

### Sushi Example

Imagine a sushi restaurant with 10,000 customers.

The restaurant has information such as:

```text
visits_per_month
average_spending
delivery_orders
restaurant_orders
customer_tenure
```

But it does not already have:

```text
customer_type
```

The restaurant wants to discover:

> What types of customer behavior appear in the data?

There is no predefined answer such as:

```text
Customer A → Frequent Customer
Customer B → Delivery Customer
Customer C → New Customer
```

K-Means tries to discover groups based on similarities in the selected numerical features.

This helped me understand the difference between supervised and unsupervised learning:

```text
SUPERVISED

Known Target
     ↓
Learn to Predict It
```

versus:

```text
UNSUPERVISED

No Known Target
     ↓
Discover Patterns or Groups
```

The features still need to be selected according to the objective of the analysis.

A feature should not be included only because it exists in the dataset.

---

#  3. How This Related to My Thesis

In my parking-demand project, I also did not have a predefined variable saying:

```text
This observation belongs to Cluster 0
This observation belongs to Cluster 1
This observation belongs to Cluster 2
```

Instead, I prepared information related to aspects such as:

```text
Driver Profile
Parking Location
Survey Zone
```

After Feature Engineering:

```text
Categorical Information
        ↓
Numerical Features
        ↓
VectorAssembler
        ↓
StandardScaler
```

and PCA:

```text
features_scaled
        ↓
PCA
        ↓
pca_features
        ↓
[PC1, PC2, PC3]
```

I used K-Means to explore groups of observations with similar numerical representations.

Therefore, the conceptual relationship is:

```text
 SUSHI

Customer Behavior
      ↓
Numerical Features
      ↓
K-Means
      ↓
Customer Groups


THESIS

Parking-Demand Information
      ↓
Numerical Features
      ↓
PCA Representation
      ↓
K-Means
      ↓
Groups of Similar Observations
```

---

# 4. What Does K Mean?

In K-Means:

```text
K = number of clusters
```

For example:

```text
K = 2 → 2 clusters
K = 3 → 3 clusters
K = 4 → 4 clusters
K = 5 → 5 clusters
```

This is different from `k` in PCA.

```text
PCA(k=3)
→ 3 Principal Components


KMeans(k=3)
→ 3 Clusters
```

The same letter is used, but it represents two completely different concepts.

This distinction was especially important in my workflow because I used:

```text
PCA
k = 3
```

to create:

```text
PC1
PC2
PC3
```

and then K-Means used those components to create clusters.

---

# 5. How Does K-Means Assign a Record to a Cluster?

This was easier for me to understand using **distances**.

Imagine that one sushi customer has the following distances from three cluster centroids:

```text
Cluster 0 → distance = 4.8
Cluster 1 → distance = 1.6
Cluster 2 → distance = 6.2
```

The smallest distance is:

```text
1.6
```

which corresponds to:

```text
Cluster 1
```

Therefore:

```text
Customer
   ↓
Closest Centroid
   ↓
Cluster 1
```

The rule I learned is:

> **K-Means assigns each record to the cluster whose centroid is closest according to the distance measure being used.**

A simple way to remember it is:

```text
SMALLEST DISTANCE
        ↓
CLOSEST CENTROID
        ↓
ASSIGNED CLUSTER
```

---

# 6. What Is a Centroid?

A centroid represents the **center of a cluster in the feature space**.

For a very simple example, imagine that the only feature is:

```text
visits_per_month
```

and three sushi customers have:

```text
Customer A → 23 visits
Customer B → 24 visits
Customer C → 25 visits
```

The mean is:

```text
(23 + 24 + 25) / 3 = 24
```

So the center is:

```text
24
```

With real Machine Learning data, each record contains several dimensions instead of one value.

Conceptually:

```text
Cluster
 │
 ├── Record A
 ├── Record B
 ├── Record C
 └── ...
        ↓
     Centroid
```

The centroid represents the center of the records assigned to that cluster in the feature space.

---

# 7. How K-Means Works

K-Means does not assign the records only once.

It is an **iterative process**.

In a simplified way:

```text
1. Initialize K centroids
        ↓
2. Calculate distances
        ↓
3. Assign each record
   to the closest centroid
        ↓
4. Recalculate the centroid
   of each cluster
        ↓
5. Calculate distances again
        ↓
6. Reassign records if needed
        ↓
7. Repeat until the algorithm
   converges or reaches a
   stopping condition
```

This also helped me understand the name:

```text
K
↓
Number of Clusters


Means
↓
Cluster Centers
Based on Means
```

---

# 8. How Did I Evaluate Different Values of K?

I learned that I do not automatically know how many clusters should be created.

One approach is to test several values.

For example:

```text
K = 2
K = 3
K = 4
K = 5
```

A `for` loop can repeat the experiment:

```python
for k in range(2, 6):
    # Train K-Means
    # Evaluate this K
```

But the `for` loop does **not** decide which K is more appropriate.

It only allows several values to be tested.

To compare the resulting clusterings, I used the **Silhouette Score**.

---

# 9. Understanding the Silhouette Score

The Silhouette Score helped me evaluate the clustering results.

I think about two questions:

```text
Are records inside the same
cluster relatively cohesive?

            +

Are different clusters
well separated?
```

A general interpretation is:

```text
Closer to 1
→ stronger cohesion and separation

Closer to 0
→ more overlap between clusters

Negative values
→ some observations may be closer
  to another cluster
```

The important distinction is:

```text
K-MEANS

Where should this record
be assigned?

       ↓

Distance to Centroids
```

while:

```text
SILHOUETTE

How cohesive and separated
is the resulting clustering?
```

---

#  10. Testing K — Sushi Example

Suppose I test:

| K | Silhouette |
|---:|---:|
| 2 | 0.38 |
| 3 | **0.67** |
| 4 | 0.55 |
| 5 | 0.49 |

Among these tested values:

```text
K = 3
```

has the highest Silhouette Score:

```text
0.67
```

Therefore, the correct interpretation is:

> Among the tested values of K, K=3 produced the highest Silhouette Score.

It does **not** mean:

```text
K = 3 is always the perfect
number of clusters
```

It only means that it produced the highest Silhouette Score among the tested configurations in this example.

---

# 11. K-Means Does Not Automatically Explain the Groups

Suppose K-Means returns:

```text
Cluster 0 → 3,000 customers
Cluster 1 → 2,500 customers
Cluster 2 → 3,500 customers
Cluster 3 → 1,000 customers
```

K-Means does not automatically tell me:

```text
Cluster 0 = Frequent Customers
Cluster 1 = New Customers
Cluster 2 = Delivery Customers
Cluster 3 = High-Spending Customers
```

K-Means only creates the groups.

To understand what each group represents, I need to return to variables that are meaningful to a person.

For example:

```text
Cluster 0

Average visits/month      → 18
Average spending          → $190,000
Average delivery orders   → 3
Average restaurant orders → 15
Average customer tenure   → 30 months
```

Based on those observed characteristics, I could describe the pattern.

For example:

> Cluster 0 contains customers with frequent visits, higher spending and more restaurant orders.

The important concept is:

```text
K-Means
   ↓
Creates Groups


Original Variables
   ↓
Help Explain Groups
```

The interpretation must come from the **actual characteristics observed in each cluster**.

---

# 12. What Happens When PCA Is Used Before K-Means?

In my workflow, PCA was applied before K-Means.

The process was:

```text
Original Data
      ↓
Feature Engineering
      ↓
StandardScaler
      ↓
PCA
      ↓
pca_features
      ↓
K-Means
      ↓
prediction
      ↓
Connect Cluster Assignment
Back to Understandable Variables
      ↓
Interpret the Groups
```

K-Means used:

```text
PC1
PC2
PC3
```

to calculate similarity and create the groups.

However, PCA coordinates are not always easy for a person to interpret.

Therefore, after clustering, I returned to descriptive variables from the original dataset to understand the characteristics of the observations in each cluster.

---

#  13. PySpark — Sushi Example

Assume PCA already produced a Delta table containing:

```text
customer_id
pca_features
```

I can load it in Databricks:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df_pca = spark.table(
    "mart_myproject.ml_features_pca"
)

df_pca.select(
    "customer_id",
    "pca_features"
).show(
    5,
    truncate=False
)
```

---

# 14. Testing Different Values of K with PySpark

I can test several K values with PySpark:

```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator(
    featuresCol="pca_features",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

results = []

for k in range(2, 6):

    kmeans = KMeans(
        k=k,
        seed=42,
        maxIter=20,
        initMode="k-means||",
        featuresCol="pca_features",
        predictionCol="prediction"
    )

    model = kmeans.fit(df_pca)

    predictions = model.transform(df_pca)

    silhouette = evaluator.evaluate(
        predictions
    )

    results.append(
        (k, silhouette)
    )

    print(
        f"K={k} | "
        f"Silhouette={silhouette:.4f}"
    )
```

Then the highest Silhouette Score among the tested configurations can be identified:

```python
best_k, best_silhouette = max(
    results,
    key=lambda x: x[1]
)

print(
    f"Best K among tested values: {best_k}"
)

print(
    f"Silhouette: {best_silhouette:.4f}"
)
```

The logic is:

```text
for loop
   ↓
Tests Several K Values

Silhouette
   ↓
Evaluates Each Clustering

max(...)
   ↓
Finds the Highest Score
Among Those Tested
```

---

# 15. Training the Selected K-Means Model

After selecting a value of K from the tested configurations:

```python
final_kmeans = KMeans(
    k=best_k,
    seed=42,
    maxIter=20,
    initMode="k-means||",
    featuresCol="pca_features",
    predictionCol="prediction"
)

final_model = final_kmeans.fit(
    df_pca
)

df_clusters = final_model.transform(
    df_pca
)
```

Now each record has a cluster assignment:

```python
df_clusters.select(
    "customer_id",
    "pca_features",
    "prediction"
).show(
    10,
    truncate=False
)
```

For example:

```text
customer_id | pca_features       | prediction
------------------------------------------------
1           | [1.2,-0.4,0.3]    | 2
2           | [-1.8,0.7,-0.2]   | 0
3           | [0.6,-2.1,1.1]    | 1
```

The new column:

```text
prediction
```

contains the cluster assigned by K-Means.

---

# 16. Interpreting the Sushi Clusters

The PCA coordinates helped K-Means create the groups, but they are not necessarily the easiest values for a person to interpret.

By keeping `customer_id`, I can connect the clustering result back to the original customer information.

```python
df_interpreted = (
    df_original
    .join(
        df_clusters.select(
            "customer_id",
            "prediction"
        ),
        on="customer_id",
        how="inner"
    )
)
```

Then I can summarize the original variables:

```python
from pyspark.sql import functions as F

cluster_summary = (
    df_interpreted
    .groupBy("prediction")
    .agg(
        F.count("*").alias(
            "customers"
        ),
        F.avg("visits_per_month").alias(
            "avg_visits"
        ),
        F.avg("average_spending").alias(
            "avg_spending"
        ),
        F.avg("delivery_orders").alias(
            "avg_delivery"
        ),
        F.avg("restaurant_orders").alias(
            "avg_restaurant"
        ),
        F.avg("customer_tenure").alias(
            "avg_tenure"
        )
    )
    .orderBy("prediction")
)

cluster_summary.show()
```

This is the stage where descriptions such as:

```text
Cluster 0
→ frequent restaurant customers

Cluster 1
→ newer customers

Cluster 2
→ customers who use delivery more often
```

could be created **only if the actual values observed in the clusters support those descriptions**.

---

# 17. Python / Scikit-learn — Same Logic

The same general clustering workflow can also be implemented with Python and Scikit-learn.

```python
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pdf = df_pca.select(
    "customer_id",
    "pca_features"
).toPandas()

X = np.vstack(
    pdf["pca_features"].apply(
        lambda x: x.toArray()
        if hasattr(x, "toArray")
        else np.asarray(x)
    )
)
```

Then several K values can be tested:

```python
results_python = []

for k in range(2, 6):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        max_iter=20,
        n_init=10
    )

    labels = model.fit_predict(X)

    silhouette = silhouette_score(
        X,
        labels,
        metric="sqeuclidean"
    )

    results_python.append(
        (k, silhouette)
    )

    print(
        f"K={k} | "
        f"Silhouette={silhouette:.4f}"
    )
```

Then:

```python
best_k_python, best_score_python = max(
    results_python,
    key=lambda x: x[1]
)

print(
    f"Best K among tested values: "
    f"{best_k_python}"
)

print(
    f"Silhouette: "
    f"{best_score_python:.4f}"
)
```

The logic is the same:

```text
Same PCA Representation
        ↓
Test Several K Values
        ↓
K-Means
        ↓
Silhouette
        ↓
Compare Results
```

---

# 18. PySpark and Scikit-learn May Number Clusters Differently

An important detail I learned is that cluster numbers are only **labels**.

For example:

```text
PySpark
Cluster 0
```

could represent a similar group to:

```text
Scikit-learn
Cluster 2
```

This is not necessarily an error.

The labels:

```text
Cluster 0
Cluster 1
Cluster 2
```

do **not** mean:

```text
Worst
Medium
Best
```

They are simply identifiers.

Also, PySpark and Scikit-learn may initialize and optimize the clustering independently.

Therefore:

```text
seed = 42
```

in PySpark and:

```text
random_state = 42
```

in Scikit-learn do not guarantee that two independent implementations will produce identical cluster IDs or assignments.

---

# 19. Validating the Same PySpark Result with Python

If I want Python to evaluate the **same clustering generated by PySpark**, I can use the assignments already produced by PySpark.

```python
pdf_validation = df_clusters.select(
    "pca_features",
    "prediction"
).toPandas()

X_validation = np.vstack(
    pdf_validation["pca_features"].apply(
        lambda x: x.toArray()
        if hasattr(x, "toArray")
        else np.asarray(x)
    )
)

labels_validation = (
    pdf_validation["prediction"]
    .to_numpy()
)

score_validation = silhouette_score(
    X_validation,
    labels_validation,
    metric="sqeuclidean"
)

print(
    f"Silhouette validation: "
    f"{score_validation:.4f}"
)
```

Now Python evaluates:

```text
Same PCA Coordinates
        +
Same Cluster Assignments
        ↓
Silhouette Validation
```

This is different from training an entirely new K-Means model in Scikit-learn.

---

#  20. Parking-Demand Project

In my parking-demand project, the ML flow documented in the previous stages was:

```text
16 Features
      ↓
StandardScaler
      ↓
PCA(k=3)
      ↓
PC1 + PC2 + PC3
      ↓
K-Means
```

Therefore, K-Means received:

```text
pca_features
=
[PC1, PC2, PC3]
```

instead of the original 10-dimensional feature vector.

This is an important distinction:

```text
Original Feature Space
      ↓
10 Dimensions

PCA Representation
      ↓
3 Dimensions

K-Means Input
      ↓
[PC1, PC2, PC3]
```

---

#  21. Testing K in the Project

For the clustering experiment documented in this project, I compared several values of K using the Silhouette Score.

The logic was:

```text
K = 2
K = 3
K = 4
K = 5
      ↓
Train K-Means
      ↓
Calculate Silhouette
      ↓
Compare Tested Results
```

The important interpretation is:

> The selected K should be described as the result of the configurations tested in the experiment, not as a universally correct number of clusters.

For example:

```text
Among the tested values of K,
the selected configuration produced
the highest Silhouette Score.
```

---

#  22. PySpark — Project Version

I loaded the PCA result produced in the previous stage:

```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

df_pca = spark.table(
    "mart_parqueo.ml_features_pca"
)
```

I configured the evaluator:

```python
evaluator = ClusteringEvaluator(
    featuresCol="pca_features",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)
```

Then I tested different values of K:

```python
results = []

for k in range(2, 6):

    kmeans = KMeans(
        k=k,
        seed=42,
        maxIter=20,
        initMode="k-means||",
        featuresCol="pca_features",
        predictionCol="prediction"
    )

    model = kmeans.fit(
        df_pca
    )

    predictions = model.transform(
        df_pca
    )

    score = evaluator.evaluate(
        predictions
    )

    results.append(
        (k, score)
    )

    print(
        f"K={k} | "
        f"Silhouette={score:.4f}"
    )
```

Then:

```python
best_k, best_score = max(
    results,
    key=lambda x: x[1]
)

print(
    f"Best K among tested values: {best_k}"
)

print(
    f"Silhouette: {best_score:.4f}"
)
```

This allowed me to compare the tested configurations using the same evaluation metric.

---

# 23. Interpreting the Thesis Clusters

After K-Means created the groups, I returned to variables that were easier to understand.

For example:

```text
Driver Profile
Parking Location
Survey Zone
```

The logic was:

```text
PCA
 ↓
Represents Records
with Fewer Dimensions


K-Means
 ↓
Creates Groups


Original Variables
 ↓
Help Explain Groups
```

The useful question is not only:

```text
prediction = 0
```

The useful question is:

> What characteristics are common among the observations assigned to Cluster 0?

This is the stage where a numerical cluster becomes an interpretable group.

---

# 24. Choosing Features Carefully

Suppose the dataset also contains an outcome such as:

```text
final_rental_decision
```

I learned that I should not automatically include a variable in K-Means simply because it exists.

The question is:

> Do I want this variable to influence how similarity between observations is defined?

If the objective is to discover groups based on profile, location, and parking behavior, a target-like outcome can be analyzed **after** clustering instead of being used to define the clusters.

Conceptually:

```text
Selected Features
       ↓
K-Means
       ↓
Clusters
       ↓
Analyze Other Variables
Inside Each Cluster
```

For example:

```text
How many observations in each cluster
show willingness to rent?
```

The lesson I learned is:

> **K-Means features should be selected according to the objective of the segmentation, not simply because the columns are available.**

---

# 25. Saving the Results in Databricks

After generating the cluster assignments, I saved the result as a Delta table:

```python
df_clusters.select(
    "id_encuesta",
    "pca_features",
    "prediction"
).write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "mart_parqueo.ml_clusters"
    )
```

The important new column is:

```text
prediction
```

which contains the cluster assigned to each observation.

The final ML pipeline was:

```text
mart_parqueo.ml_features_scaled
        ↓
PCA
        ↓
mart_parqueo.ml_features_pca
        ↓
K-Means
        ↓
mart_parqueo.ml_clusters
```

---

# 26. Sushi Example vs. My Thesis

The relationship between the learning example and my implementation can be summarized as:

```text
SUSHI EXAMPLE

Customer Data
      ↓
Feature Engineering
      ↓
StandardScaler
      ↓
PCA
      ↓
K-Means
      ↓
Customer Groups
      ↓
Interpret Original
Customer Variables
```

and:

```text
 MY THESIS

Parking-Demand Data
      ↓
Feature Engineering
      ↓
16 Features
      ↓
StandardScaler
      ↓
PCA (k=3)
      ↓
[PC1, PC2, PC3]
      ↓
K-Means
      ↓
prediction
      ↓
Interpret Original
Parking-Demand Variables
```

The datasets are different, but the Machine Learning reasoning is similar.

---

# What I Learned

Through this implementation I learned that:

- K-Means is an **unsupervised learning algorithm**.
- K-Means creates groups based on similarity in the selected feature space.
- `K` in K-Means represents the number of clusters.
- `k` in PCA and `k` in K-Means represent different concepts.
- K-Means assigns observations according to their distance from cluster centroids.
- A centroid represents the center of a cluster in the feature space.
- K-Means is iterative: assignments and centroids are recalculated during training.
- I can test several K values instead of assuming the number of clusters in advance.
- The Silhouette Score can help compare clustering results.
- A higher Silhouette Score among tested configurations does not prove that the selected K is universally optimal.
- Cluster numbers are only identifiers.
- K-Means creates the groups, but the original variables help me interpret them.
- PCA is not mandatory before K-Means.
- In my implementation, I used `[PC1, PC2, PC3]` as the input for K-Means.
- Feature selection should follow the objective of the analysis.
- Saving intermediate Delta tables helped me inspect and validate each stage independently.

The three concepts I needed to keep separate were:

```text
PCA
→ reduces dimensionality


K-MEANS
→ groups similar records using
  distances to centroids


SILHOUETTE
→ evaluates cohesion inside clusters
  and separation between clusters
```

And the two rules that helped me the most were:

```text
Assign a Record
      ↓
Closest Centroid
      ↓
Cluster Assignment
```

and:

```text
Compare Tested K Values
      ↓
Silhouette Score
      ↓
Evaluate the Resulting Clusterings
```

---

# Complete Machine Learning Flow

The complete process documented in these learning notes is:

```text
Parking-Demand Data
        ↓
Feature Engineering
        ↓
10 Features
        ↓
VectorAssembler
        ↓
features_raw
        ↓
StandardScaler
        ↓
features_scaled
        ↓
PCA (k=3)
        ↓
pca_features
[PC1, PC2, PC3]
        ↓
K-Means
        ↓
prediction
        ↓
Cluster Interpretation
        ↓
mart_parqueo.ml_clusters
```

---

# Important Note

This document describes a **learning implementation based on my academic project**.

The sushi examples are simplified examples used to explain the concepts.

The parking-demand sections describe how I applied the Machine Learning workflow to my thesis data.

K-Means results should always be interpreted in the context of:

```text
Selected Features
Data Quality
Scaling
Dimensionality Reduction
Chosen K Values
Evaluation Metric
Dataset Size
Analysis Objective
```

A clustering result does not automatically prove that the discovered groups represent fixed or universal categories in the real world.

The purpose of this exercise was to learn how to build, evaluate, and interpret an unsupervised Machine Learning workflow using PySpark and Databricks.

---

⬅️ [Back to Machine Learning — PCA](Machine_Learning_PCA.md)

⬅️ [Back to main README](README.md)

# K-Means Clustering

These are my notes about **K-Means**, written in the same simple way I learned the concept.

The complete flow is:

```text
Feature Engineering
        ↓
StandardScaler
        ↓
PCA (optional)
        ↓
K-Means
        ↓
Silhouette
        ↓
Interpret the clusters
```

---

# Before K-Means: Do I Always Need PCA?

No. **PCA is not mandatory before K-Means.**

This is important because I can have either:

```text
Feature Engineering
        ↓
StandardScaler
        ↓
K-Means
```

or:

```text
Feature Engineering
        ↓
StandardScaler
        ↓
PCA
        ↓
K-Means
```

## Gym Example

Imagine a gym wants to discover groups of customers with similar behavior.

If I only have five features:

```text
Visits per Month
Months as Customer
Monthly Spending
Group Classes per Month
Uses Personal Trainer
```

I may not need PCA.

I only have five dimensions, so after `StandardScaler` I can try K-Means directly.

Now imagine I have **25 features**.

In that case, testing PCA may be more useful.

Suppose I apply:

```text
PCA(k=4)
```

and obtain:

```text
PC1 → 28%
PC2 → 19%
PC3 → 13%
PC4 → 10%

Cumulative Explained Variance → 70%
```

This means:

```text
25 original features
        ↓
4 Principal Components
        ↓
70% of the original variability represented
```

It does **not** mean 70% accuracy.

There is a trade-off:

```text
Benefit:
25 dimensions → 4 dimensions

Cost:
30% of the original variability is not represented
```

So my decision is not:

> I must use PCA because I am going to use K-Means.

Instead:

> I can test PCA, see how much I reduce the dimensionality and how much cumulative variance I preserve, and then decide whether using PCA before K-Means makes sense.

I can also compare:

```text
OPTION A

StandardScaler
      ↓
K-Means
```

with:

```text
OPTION B

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
Can I represent this information with fewer dimensions?

K-Means
↓
Which records have similar behavior?
```

---

# What Is K-Means?

K-Means is an **unsupervised Machine Learning algorithm** that creates groups of similar records.

It is unsupervised because I do not have a target that tells the model the correct group for each record.

## Sushi Example

Imagine a sushi restaurant with 10,000 customers.

I have:

```text
visits_per_month
average_spending
delivery_orders
restaurant_orders
customer_tenure
```

But I do not have:

```text
customer_type
```

The restaurant wants to discover:

> What types of customers do we have?

I do not already know the answer.

That is why this is an **unsupervised learning problem**.

A simple way I remember the difference is:

```text
SUPERVISED

I have a known target
        ↓
I want to predict it
```

```text
UNSUPERVISED

I do not have a known target
        ↓
I want to discover patterns or groups
```

The features are selected because each one gives information about the behavior I want to study.

I do not include a feature only because it exists.

---

# What Does K Mean?

In K-Means:

```text
K = number of clusters I want to create
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
→ 3 clusters
```

The same letter is used, but it means two different things.

---

# How Does K-Means Decide the Cluster?

This was easier for me to understand using **distances**.

Suppose one customer has these distances:

```text
Cluster 0 → distance = 4.8
Cluster 1 → distance = 1.6
Cluster 2 → distance = 6.2
```

The smallest distance is:

```text
1.6
```

and it belongs to:

```text
Cluster 1
```

Therefore:

```text
Customer → Cluster 1
```

The rule I need to remember is:

> **K-Means assigns each record to the cluster whose centroid has the smallest distance.**

Or even shorter:

```text
SMALLEST DISTANCE
        ↓
CLOSEST CENTROID
        ↓
THAT CLUSTER
```

---

# What Is a Centroid?

A centroid represents the **center of a cluster**.

For a very simple example, imagine I only have one feature:

```text
visits_per_month
```

and three customers:

```text
Customer A → 23 visits
Customer B → 24 visits
Customer C → 25 visits
```

The center is:

```text
(23 + 24 + 25) / 3 = 24
```

So the centroid is approximately:

```text
24
```

With real data, I have several dimensions instead of only one number.

But the basic idea is the same:

> The centroid represents the center of the records assigned to that cluster.

---

# How K-Means Works

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
7. Repeat until the solution stabilizes
```

This also helps me understand the name:

```text
K
↓
number of clusters

Means
↓
the centers are based on means
```

---

# How Do I Choose K?

I do not automatically know how many clusters I should use.

One approach is to test several values.

For example:

```text
K = 2
K = 3
K = 4
K = 5
```

A `for` loop lets me repeat the experiment:

```python
for k in range(2, 6):
    # Train K-Means
    # Evaluate this K
```

But the `for` does **not** decide which K is better.

It only allows me to test several values.

To compare the results, I need a metric.

One metric I can use is the **Silhouette Score**.

---

# Silhouette Score

The Silhouette Score helps me evaluate how the clustering turned out.

I think about two questions:

```text
Are the records inside the same cluster cohesive / close to each other?

              +

Are the different clusters well separated from each other?
```

A simple interpretation is:

```text
Closer to 1
→ better cohesion inside clusters and better separation between clusters

Closer to 0
→ more overlap between clusters

Negative
→ some records may fit another  cluster better
```

The important distinction is:

```text
K-MEANS

Which cluster should this record belong to?

        ↓

Look for the SMALLEST distance to a centroid
```

while:

```text
SILHOUETTE

How good is the resulting grouping?

        ↓

When comparing tested K values, a HIGHER score is generally preferred
```

## Sushi Example

Suppose I test:

```text
K = 2 → Silhouette = 0.38
K = 3 → Silhouette = 0.67
K = 4 → Silhouette = 0.55
K = 5 → Silhouette = 0.49
```

Among these tested values:

```text
K = 3
```

has the highest Silhouette Score:

```text
0.67
```

Therefore, I can say:

> Among the tested values of K, K=3 produced the highest Silhouette Score.

I should **not** say that 3 is universally the perfect number of clusters.

I only know that it was the best result **among the values I tested according to this metric**.

---

# K-Means Does Not Name the Groups

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

K-Means only created the groups.

To understand what each group represents, I return to the **original variables**.

For example:

```text
Cluster 0

Average visits/month       → 18
Average spending           → $190,000
Average delivery orders    → 3
Average restaurant orders  → 15
Average customer tenure    → 30 months
```

Now I can interpret the pattern.

For example:

> Cluster 0 contains customers with frequent visits, higher spending and more restaurant orders.

The important idea is:

```text
K-Means
↓
creates the groups

Original variables
↓
help me understand the groups
```

---

# What Happens If I Used PCA?

If I used PCA before K-Means, the process is:

```text
Original Data
      ↓
Feature Engineering
      ↓
StandardScaler
      ↓
PCA
      ↓
K-Means
      ↓
prediction / cluster
      ↓
Connect the cluster assignment
back to the original data
      ↓
Interpret each group
```

K-Means can work with:

```text
PC1
PC2
PC3
```

to create the groups.

But when I want to understand what those groups mean, I can return to variables that are easier for me to interpret.

---

# PySpark — Sushi Example

Assume PCA already produced a Delta table containing:

```text
customer_id
pca_features
```

I can load it in Databricks:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df_pca = spark.table("mart_myproject.ml_features_pca")

df_pca.select(
    "customer_id",
    "pca_features"
).show(5, truncate=False)
```

---

# Testing Different Values of K with PySpark

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

Then I can find the highest Silhouette among the tested values:

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
→ tests several K values

Silhouette
→ evaluates each result

max(...)
→ finds the highest score
  among the tested values
```

---

# Training the Final K-Means Model

Once I select `best_k`:

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

Now each customer has a cluster assignment:

```python
df_clusters.select(
    "customer_id",
    "pca_features",
    "prediction"
).show(10, truncate=False)
```

For example:

```text
customer_id | pca_features        | prediction
------------------------------------------------
1           | [1.2, -0.4, 0.3]   | 2
2           | [-1.8, 0.7, -0.2]  | 0
3           | [0.6, -2.1, 1.1]   | 1
```

`prediction` is the cluster assigned by K-Means.

---

# Interpreting the Sushi Clusters

The PCA coordinates helped K-Means create the groups, but they are not always the easiest values for a person to interpret.

I can keep `customer_id` and connect the result back to the original customer data.

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

Then I can calculate summaries using the original variables:

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

This is where I can start saying things like:

```text
Cluster 0
→ frequent restaurant customers

Cluster 1
→ newer customers

Cluster 2
→ customers who use delivery more often
```

But these descriptions must come from the **actual values observed in each cluster**.

---

# Python / scikit-learn — Same Logic

I can reproduce the same workflow using Python and scikit-learn.

First I use the same PCA representation:

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

Then I test several values of K:

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
same PCA representation
        ↓
test several K values
        ↓
K-Means
        ↓
Silhouette
        ↓
compare the results
```

---

# Important: PySpark and Python May Number Clusters Differently

For example:

```text
PySpark
Cluster 0
```

could represent the same group that Python calls:

```text
scikit-learn
Cluster 2
```

This is not necessarily an error.

The cluster numbers are only **labels**.

```text
Cluster 0
Cluster 1
Cluster 2
```

does not mean:

```text
worst
medium
best
```

They are simply identifiers.

Also, PySpark and scikit-learn may initialize the centroids differently.

Because of that, using:

```text
seed = 42
```

in PySpark and:

```text
random_state = 42
```

in scikit-learn does not guarantee that both independent executions will produce exactly the same cluster IDs or assignments.

---

# Validating the Same PySpark Result with Python

If I want Python to evaluate the **same clustering generated by PySpark**, I can take the assignments already produced by PySpark.

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

Now Python is evaluating:

```text
same PCA coordinates
+
same cluster assignments
```

produced by PySpark.

---

# Practical Result — Parking Demand Project

In my parking-demand project, the unsupervised flow was:

```text
16 features
      ↓
StandardScaler
      ↓
PCA(k=3)
      ↓
PC1 + PC2 + PC3
      ↓
K-Means
```

The PCA result was:

```text
PC1 → 20.48%
PC2 → 15.06%
PC3 → 13.00%

Cumulative Explained Variance
→ 48.54%
```

Therefore, K-Means received:

```text
pca_features
=
[PC1, PC2, PC3]
```

instead of the original 16-dimensional vector.

The final clustering dataset contained:

```text
90 unique survey observations
```

---

# Testing K in the Project

I tested:

| K | Silhouette |
|---:|---:|
| 2 | 0.5585 |
| 3 | 0.5044 |
| 4 | 0.6539 |
| 5 | **0.7211** |

Among the tested values:

```text
K = 5
```

had the highest Silhouette Score:

```text
0.7211
```

Therefore, I selected:

```text
K = 5
```

for the final segmentation.

The correct way for me to explain this is:

> Among K=2, K=3, K=4 and K=5, K=5 produced the highest Silhouette Score.

It does not mean that five clusters are universally the only possible solution.

---

# Final Cluster Distribution

The 90 observations were distributed as:

| Cluster | Records | Percentage |
|---:|---:|---:|
| 0 | 29 | 32.22% |
| 1 | 29 | 32.22% |
| 2 | 22 | 24.44% |
| 3 | 4 | 4.44% |
| 4 | 6 | 6.67% |

Again:

```text
Cluster 4
```

does not mean it is better than:

```text
Cluster 1
```

The numbers are only identifiers.

---

# PySpark — Project Version

```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

df_pca = spark.table(
    "mart_parqueo.ml_features_pca"
)

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

    score = evaluator.evaluate( predictions )

    results.append( (k, score) )

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
    f"Best K: {best_k}"
)

print(
    f"Silhouette: {best_score:.4f}"
)
```

For the final project:

```text
Best K among tested values → 5
Silhouette                 → 0.7211
```

Then I trained the final model:

```python
kmeans_final = KMeans(
    k=5,
    seed=42,
    maxIter=20,
    initMode="k-means||",
    featuresCol="pca_features",
    predictionCol="prediction"
)

kmeans_model = kmeans_final.fit(
    df_pca
)

df_clusters = kmeans_model.transform(
    df_pca
)
```

---

# Python Validation — Project

I can validate the PySpark clustering using Python with the same assignments:

```python
import numpy as np

from sklearn.metrics import silhouette_score

pdf_clusters = df_clusters.select(
    "id_encuesta",
    "pca_features",
    "prediction"
).toPandas()

X = np.vstack(
    pdf_clusters["pca_features"].apply(
        lambda x: x.toArray()
        if hasattr(x, "toArray")
        else np.asarray(x)
    )
)

labels = (
    pdf_clusters["prediction"]
    .to_numpy()
)

silhouette_python = silhouette_score(
    X,
    labels,
    metric="sqeuclidean"
)

print(
    f"Silhouette calculated in Python: "
    f"{silhouette_python:.4f}"
)
```

The purpose here is to evaluate the **same clustering** generated in PySpark.

---

# Interpreting the Project Clusters

After K-Means created the groups, I returned to variables that are easier to understand.

For example:

```text
driver profile
geographic zone
willingness to rent
```

The logic is:

```text
PCA
→ represents the records with fewer dimensions

K-Means
→ creates the groups

Original variables
→ help explain the groups
```

The useful question is not only:

```text
prediction = 0
```

The useful question is:

> What characteristics are common among the observations assigned to Cluster 0?

That is how the cluster becomes understandable.

---

# Choosing Features Carefully

Suppose I also have:

```text
final_rental_decision
```

I should not automatically add it to K-Means just because the column exists.

First I ask:

> Do I want this variable to influence how similarity between records is defined?

If my objective is to discover groups based on profile, location and parking/traffic behavior, I can leave a target-like outcome outside the clustering features.

Then I can analyze it **after** K-Means:

```text
K-Means creates clusters
        ↓
Then I analyze:

How many people in each cluster
show willingness to rent?
```

The lesson is:

> **I choose K-Means features according to the objective of the segmentation, not simply because the columns are available.**

---

# Saving the Results in Databricks

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

---

The three concepts I need to keep separate are:

```text
PCA
→ reduces dimensionality

K-MEANS
→ groups similar records using
  distances to centroids

SILHOUETTE
→ evaluates cohesion inside clusters  and separation between clusters
```

And the two rules that helped me the most are:

```text
Assign a record to a cluster
→ SMALLEST distance to the centroid
```

```text
Compare different K values
→ generally look for the HIGHEST
  Silhouette Score among those tested
```


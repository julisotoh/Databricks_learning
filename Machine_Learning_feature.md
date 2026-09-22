# Machine Learning — Feature Engineering with PySpark

This document explains the **Feature Engineering process I implemented with PySpark and Spark ML** as part of my Master's thesis.

The objective of this stage was to transform and prepare information from the parking-demand dataset so it could later be used for **PCA and K-Means clustering**.

To make some concepts easier to understand, I also use a simple **sushi example** and then connect the concept with the implementation used in my thesis.

> **Learning note**
>
> This repository documents what I implemented and learned while working with Databricks, PySpark, and Spark ML.
>
> The sushi examples are simplified learning examples. The parking-demand examples show how I applied these concepts in my academic project.

---

# What I Learned

## 1. Preparing the Dataset

The first thing I learned was that having clean data does not mean that the data is already prepared for Machine Learning.

First, I needed to understand the objective of the analysis and retrieve the information that could help me reach that objective.

Depending on the data model, this may require joining Fact and Dimension tables, filtering records, selecting columns, aggregating information, or performing other transformations.

For example:

```python
from pyspark.sql import functions as F

df_base = (
    fact_data.alias("f")
    .join(
        dim_category.alias("c"),
        F.col("f.category_id") == F.col("c.category_id"),
        "left"
    )
    .select(
        "f.record_id",
        "c.category",
        "f.amount",
        "f.quantity"
    )
)
```

The result is a base DataFrame that can be used to start preparing the features.

### In my thesis

For my Machine Learning dataset, I combined information from the survey fact table with dimensions containing information about the driver profile and location.

Conceptually:

```text
fact_encuesta
      │
      ├──── dim_perfil_conductor
      │
      └──── dim_ubicacion
              │
              ▼
         Base ML Dataset
```

This allowed me to bring together information such as:

```text
Survey ID
Driver Profile
Parking Location
Survey Zone
```

before creating the ML features.

---

## 2. Understanding the Variables

Before transforming the data, I needed to understand what type of variables I had.

Some common types are:

| Variable type | What it means | Example |
|---|---|---|
| **Numerical** | Represents a measurable or countable value | Price, quantity, speed |
| **Categorical** | Represents a group or category | Location, profile, vehicle type |
| **Binary** | Has only two possible states | Yes/No, 1/0 |
| **Date/Time** | Represents a date or time | Created date, hour, month |

This distinction is important because not every variable can be sent to a Machine Learning algorithm in its original format.

For example, numerical information can already be represented as numbers, while categorical information such as:

```text
Home
Work
Both
```

may need to be transformed before it can be used by the algorithm.

---

## 3. Transforming Categorical Variables

One of the transformations I used was converting categorical information into numerical information.

### Simple Example — Sushi Preferences

Imagine a dataset containing:

| Person | Favorite Sushi |
|---|---|
| Ana | Salmon |
| John | Tuna |
| Laura | California Roll |
| David | Salmon |

`Favorite Sushi` is a categorical variable.

A possible mistake would be to assign arbitrary numbers:

```text
Salmon          = 1
Tuna            = 2
California Roll = 3
```

This could introduce an artificial numerical relationship between categories.

Instead, the categories can be represented using binary columns:

| Person | Salmon | Tuna | California Roll |
|---|---:|---:|---:|
| Ana | 1 | 0 | 0 |
| John | 0 | 1 | 0 |
| Laura | 0 | 0 | 1 |
| David | 1 | 0 | 0 |

For example:

```text
Favorite Sushi = Salmon

Salmon          = 1
Tuna            = 0
California Roll = 0

        ↓

[1, 0, 0]
```

This type of representation is known as **One-Hot Encoding**.

Each category is represented by a binary column:

- `1` means that the record belongs to that category.
- `0` means that it does not.

### How I applied the concept in my thesis

My thesis dataset contained categorical information such as:

```text
Driver Profile
Parking Location
Survey Zone
```

For example, some driver profiles were:

```text
Tiene parqueadero disponible para rentar
Busca parqueadero
Tiene parqueadero pero busca afuera
```

For the clustering exercise, I created binary features such as:

| Driver Profile | Active Owner | Demand | Passive Owner |
|---|---:|---:|---:|
| Tiene parqueadero disponible para rentar | 1 | 0 | 0 |
| Busca parqueadero | 0 | 1 | 0 |
| Tiene parqueadero pero busca afuera | 0 | 0 | 1 |

Conceptually:

```text
 Sushi Example

"Salmon"
    │
    ▼
[1, 0, 0]


 Thesis Example

"Busca parqueadero"
    │
    ▼
[0, 1, 0]
```

In my implementation, I created these binary features manually using PySpark conditions:

```python
from pyspark.sql import functions as F

df_features = (
    df_base
    .withColumn(
        "profile_active_owner",
        F.when(
            F.col("driver_profile") ==
            "Tiene parqueadero disponible para rentar",
            1
        ).otherwise(0)
    )
    .withColumn(
        "profile_demand",
        F.when(
            F.col("driver_profile") ==
            "Busca parqueadero",
            1
        ).otherwise(0)
    )
    .withColumn(
        "profile_passive_owner",
        F.when(
            F.col("driver_profile") ==
            "Tiene parqueadero pero busca afuera",
            1
        ).otherwise(0)
    )
)
```

I applied the same idea to parking location:

```text
En la casa
En el trabajo
En ambos lugares
```

and survey zones.

For this exercise, I created the binary columns manually using `when()`.

PySpark also provides tools such as `StringIndexer` and `OneHotEncoder` that can be used for categorical encoding depending on the problem and implementation.

> **What I learned:**  
> The important part is not only converting text into numbers. I also need to make sure that the numerical representation does not introduce relationships that do not exist in the original categories.

---

## 4. Selecting the Features

One of the most important things I learned is that the features depend on the objective of the analysis.

Not every column in a DataFrame needs to be used as a feature.

I first needed to understand what I wanted to analyze and then select existing variables or create new variables that provided useful information for that objective.

### Example — Predicting Sushi Demand

Imagine that I have a restaurant database and want to estimate sushi demand for the next month.

My source data could contain:

```text
customer_id
order_date
product
quantity
delivery_type
price
```

I would not automatically use all these columns as features.

For example:

```text
customer_id
```

identifies a customer, but the ID itself does not describe customer behavior.

However, the ID could be used to calculate something more useful:

```text
customer_id
      ↓
COUNT previous sushi orders
      ↓
sushi_orders_per_customer
      ↓
FEATURE
```

Depending on the objective and the available historical data, useful features could include:

```text
sushi_orders
recurring_customers
sushi_deliveries
average_price
```

The important concept is:

> A feature is not simply a column that exists in a DataFrame. It should represent useful information related to the objective of the analysis.

An identifier can therefore be useful for creating a feature even when the identifier itself is not a useful feature.

### Feature selection in my thesis

In my thesis, I selected features related to:

```text
Driver Profile
Parking Location
Survey Zone
```

The feature set used for this clustering exercise contained:

```python
feature_cols = [
    "profile_active_owner",
    "profile_demand",
    "profile_passive_owner",
    "location_home",
    "location_work",
    "location_both",
    "zone_south",
    "zone_center",
    "zone_north",
    "zone_west"
]
```

These were the features I prepared for the PCA and K-Means stages of the exercise.

The sushi example and my thesis use different data, but the reasoning is similar:

```text
Business / Analysis Objective
            ↓
Understand the available data
            ↓
Select or create useful information
            ↓
Features
```

---

## 5. Handling Missing Values

Before creating the feature vector, I also needed to check whether the selected features contained missing values.

In my implementation, some numerical variables did not have information available for every record.

Instead of leaving these values as `NULL`, I calculated the median of each variable and used it to replace the missing values where this treatment was appropriate.

For example:

```python
median_value = df_base.select(
    F.expr(
        "percentile_approx(numerical_feature, 0.5)"
    ).alias("median_value")
).collect()[0]["median_value"]

df_base = df_base.withColumn(
    "numerical_feature",
    F.coalesce(
        F.col("numerical_feature"),
        F.lit(median_value)
    )
)
```

Conceptually:

```text
Original values

30
35
NULL
40
45

      ↓ Replace NULL with the median

30
35
40
40
45
```

This process is known as **missing value imputation**.

I learned that the median can be useful for numerical data because it is less affected by extreme values than the mean.

The objective here is different from standardizing the features:

```text
Median
   ↓
Used to replace missing values


StandardScaler
   ↓
Used later to standardize
the scale of the features
```

> **Important:** Missing values should not always be replaced automatically. The treatment depends on what the missing value means and on the characteristics of the dataset.

---

## 6. Creating the Feature Vector with VectorAssembler

After selecting and preparing the features, they were still stored as individual columns in the DataFrame.

###  Sushi Example

Imagine that the selected features are:

| sushi_orders | recurring_customers | sushi_deliveries | average_price |
|---:|---:|---:|---:|
| 320 | 85 | 120 | 42.5 |
| 350 | 91 | 140 | 43.2 |
| 410 | 110 | 170 | 41.8 |

First, the feature columns are defined:

```python
feature_cols = [
    "sushi_orders",
    "recurring_customers",
    "sushi_deliveries",
    "average_price"
]
```

PySpark Machine Learning expects the input features to be grouped into a vector.

For this, I used `VectorAssembler`:

```python
from pyspark.ml.feature import VectorAssembler

vector_assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

df_vectorized = vector_assembler.transform(df_features)
```

`VectorAssembler` does not train a model and does not change the original values.

It groups the selected features into a single vector.

For example:

```text
sushi_orders          = 320
recurring_customers   = 85
sushi_deliveries      = 120
average_price         = 42.5

                 ↓ VectorAssembler

features_raw = [320, 85, 120, 42.5]
```

### In my thesis

I applied the same concept to the 10 features selected for the clustering exercise:

```text
profile_active_owner ─┐
profile_demand ───────┤
profile_passive_owner ┤
location_home ────────┤
location_work ────────┤
location_both ────────┤
zone_south ───────────┤
zone_center ──────────┤──► VectorAssembler
zone_north ───────────┤
zone_west ────────────┘
                               │
                               ▼
                          features_raw
```

In PySpark:

```python
from pyspark.ml.feature import VectorAssembler

vector_assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

df_vectorized = vector_assembler.transform(df_features)
```

This was an important concept for me:

```text
Several Feature Columns
          ↓
    VectorAssembler
          ↓
   One Feature Vector
```

The resulting `features_raw` vector could then be used by other PySpark ML transformations.

---

## 7. Standardizing the Features with StandardScaler

After creating the feature vector, another problem can appear: the features may have very different numerical scales.

###  Sushi Example

For example:

```text
is_delivery       = 1
is_weekend        = 0
average_price     = 42.5
sushi_orders      = 320
total_sales       = 15000
```

After `VectorAssembler`, the vector could look like:

```text
features_raw = [1, 0, 42.5, 320, 15000]
```

These values are on very different scales.

For algorithms that are sensitive to scale, a variable should not dominate the analysis only because its numerical values are much larger than the others.

For this reason, I used `StandardScaler`.

```python
from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(df_vectorized)

df_scaled = scaler_model.transform(df_vectorized)
```

With `withMean=True` and `withStd=True`, each feature is standardized using its mean and standard deviation.

Conceptually:

```text
                       value - mean
standardized value = --------------------
                     standard deviation
```

The purpose is **not** to convert every value to a range between 0 and 1.

Instead, the purpose is to put features with different numerical scales on a more comparable scale.

```text
Before scaling

binary feature     → 0 / 1
price              → tens
orders             → hundreds
sales              → thousands

             ↓ StandardScaler

features_scaled
```

### In my thesis

I also used `StandardScaler` after creating the feature vector:

```text
Thesis Features
      ↓
VectorAssembler
      ↓
features_raw
      ↓
StandardScaler
      ↓
features_scaled
```

This was particularly relevant because the resulting features were later used as input for **PCA and K-Means**, which are sensitive to the scale of the input variables.

The resulting standardized vector was stored in:

```text
features_scaled
```

---

## 8. Saving the Prepared Features

After preparing and standardizing the features, I saved the resulting DataFrame as a Delta table.

In my thesis implementation, the resulting table was:

```text
mart_parqueo.ml_features_scaled
```

Conceptually:

```python
df_scaled.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("mart_parqueo.ml_features_scaled")
```

I separated the Machine Learning process into different stages:

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
        ↓
mart_parqueo.ml_clusters
```

Saving the intermediate dataset is not a requirement for PCA or K-Means.

In my implementation, it was a way to organize the process so that each stage could be developed, executed, inspected, and validated separately.

---

## What I Learned from the Feature Engineering Stage

This stage helped me understand that Feature Engineering is not simply converting columns into numbers.

I learned that I first need to understand:

```text
What is the objective?

What information is available?

What does each variable represent?

Which information could be useful?

Does the data contain missing values?

Do categorical variables need to be transformed?

Are the numerical features on very different scales?

What format does the next ML algorithm expect?
```

I also learned the role of each PySpark component:

| Component | What I understood |
|---|---|
| `when()` | Can be used to create conditional/binary features |
| `VectorAssembler` | Groups several numerical features into one ML vector |
| `StandardScaler` | Standardizes features to a comparable scale |
| Delta table | Allowed me to persist the prepared dataset between ML stages |

The sushi examples helped me understand the concepts with simple data, while the thesis implementation showed me how I applied the same ideas to a real academic dataset.

---

## Complete Feature Engineering Flow

The complete process I followed can be summarized as:

```text
Gold / MART
      ↓
Prepare the DataFrame
      ↓
Understand the objective
      ↓
Understand the variables
      ↓
Select or create useful features
      ↓
Handle missing values
      ↓
Transform categorical variables when necessary
      ↓
VectorAssembler
      ↓
features_raw
      ↓
StandardScaler
      ↓
features_scaled
      ↓
Save prepared features
      ↓
PCA
```

This completed the Feature Engineering stage of my Machine Learning workflow.

---

## Important Note

This is a **learning implementation based on my academic project**.

The feature selection, categorical transformations, missing-value treatment, and scaling decisions documented here correspond to the dataset and objective I worked with.

They should not be interpreted as a universal Feature Engineering strategy.

The main purpose of this document is to explain **what I implemented, why I implemented it, and what I learned from the process**.

---

## Next Stage

After preparing and scaling the features, I applied **Principal Component Analysis (PCA)** for dimensionality reduction before K-Means clustering.

➡️ [Machine Learning — PCA](Machine_Learning_PCA.md)

---

⬅️ [Back to main README](README.md)

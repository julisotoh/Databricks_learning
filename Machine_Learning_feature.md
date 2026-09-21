# Feature Engineering in PySpark

## What I Learned

Before applying a Machine Learning algorithm, the first step is to prepare
the data that will be used by the model.

In my case, I started with data that had already been cleaned and transformed
through the Medallion Architecture. I used Fact and Dimension tables from
the Gold / MART layer.

```text
Gold / MART
    │
    ├── Fact Tables
    └── Dimension Tables
            │
            ▼
      Prepare Dataset
            │
            ▼
     Feature Engineering
```

---

## 1. Preparing the Dataset

The first thing I learned was that having clean data does not mean that the
data is already prepared for Machine Learning.

First, I need to understand the objective of the analysis and then retrieve
the information that can help me reach that objective.

Depending on the data model, this may require joining Fact and Dimension
tables, filtering records, selecting columns, aggregating information, or
performing other transformations.

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

The result is the base DataFrame that I will use to start preparing the
features.

---

## 2. Understanding the Variables

Before transforming the data, I need to understand what type of variables
I have.

Some common types are:

| Variable type | What it means | Example |
|---|---|---|
| **Numerical** | Represents a measurable or countable value | Price, quantity, speed |
| **Categorical** | Represents a group or category | Location, profile, vehicle type |
| **Binary** | Has only two possible states | Yes/No, 1/0 |
| **Date/Time** | Represents a date or time | Created date, hour, month |

This distinction is important because not every variable can be sent to a
Machine Learning algorithm in its original format.

For example, numerical information can already be represented as numbers,
while categorical information such as `Home`, `Work`, or `Both` may need
to be transformed before it can be used by the model.

---

## 3. Transforming Categorical Variables

One of the transformations I used was converting categorical information
into numerical information.

For example, suppose I have this categorical variable:

```text
location

Home
Work
Both
```

Instead of using the text directly, I can represent each category using
binary values:

| location | location_home | location_work | location_both |
|---|---:|---:|---:|
| Home | 1 | 0 | 0 |
| Work | 0 | 1 | 0 |
| Both | 0 | 0 | 1 |

This type of representation is known as **One-Hot Encoding**.

Each category is represented by a binary column:

- `1` means that the record belongs to that category.
- `0` means that it does not.

In my implementation, I created these binary features manually using
PySpark conditions:

```python
from pyspark.sql import functions as F

df_features = (
    df_base
    .withColumn(
        "location_home",
        F.when(F.col("location") == "Home", 1).otherwise(0)
    )
    .withColumn(
        "location_work",
        F.when(F.col("location") == "Work", 1).otherwise(0)
    )
    .withColumn(
        "location_both",
        F.when(F.col("location") == "Both", 1).otherwise(0)
    )
)
```

This allowed me to transform categorical information into numerical
features that could later be used as input for Machine Learning.

In this example, I created the binary columns manually using `when()`.
PySpark also provides `OneHotEncoder` for encoding categorical variables.

---

## 4. Selecting the Features

One of the most important things I learned is that the features depend on
the objective of the analysis.

Not every column in a DataFrame needs to be used as a feature.

First, I need to understand what I want to analyze or predict. Then I can
select existing variables or create new variables that provide useful
information for that objective.

For example, imagine that I have a restaurant database and I want to
estimate sushi demand for the next month.

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

For example, `customer_id` identifies a customer, but the ID itself does
not describe customer behavior.

However, I can use that column to calculate something more useful:

```text
customer_id
      ↓
COUNT previous sushi orders
      ↓
sushi_orders_per_customer
      ↓
FEATURE
```

Depending on the objective and the available historical data, I could
create features such as:

```text
sushi_orders
recurring_customers
sushi_deliveries
average_price
```

The important lesson for me was:

> A feature is not simply a column that exists in my DataFrame. It should
> represent useful information related to the objective of my analysis.

An identifier can therefore be useful for creating a feature even when the
identifier itself is not a useful feature.

---

## 5. Handling Missing Values

Before creating the feature vector, I also need to check whether the selected
features contain missing values.

In my implementation, some numerical variables did not have information
available for every record.

Instead of leaving these values as `NULL`, I calculated the median of each
variable and used it to replace the missing values.

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

I learned that the median can be useful for numerical data because it is
less affected by extreme values than the mean.

The objective here is different from standardizing the features.

```text
Median
   ↓
Used to replace missing values

StandardScaler
   ↓
Used later to standardize the scale of the features
```

---

## 6. Creating the Feature Vector with VectorAssembler

After selecting and preparing the features, they are still stored as
individual columns in the DataFrame.

Using the sushi example:

| sushi_orders | recurring_customers | sushi_deliveries | average_price |
|---:|---:|---:|---:|
| 320 | 85 | 120 | 42.5 |
| 350 | 91 | 140 | 43.2 |
| 410 | 110 | 170 | 41.8 |

First, I define which columns I want to use as features:

```python
feature_cols = [
    "sushi_orders",
    "recurring_customers",
    "sushi_deliveries",
    "average_price"
]
```

PySpark Machine Learning expects the input features to be grouped into a
vector.

For this, I used `VectorAssembler`:

```python
from pyspark.ml.feature import VectorAssembler

vector_assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

df_vectorized = vector_assembler.transform(df_features)
```

`VectorAssembler` does not train a model and does not change the original
values.

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

This was an important concept for me:

```text
Several feature columns
          ↓
    VectorAssembler
          ↓
One feature vector
```

The resulting `features_raw` vector can then be used by PySpark ML
transformations and algorithms.

---

## 7. Standardizing the Features with StandardScaler

After creating the feature vector, I learned that another problem can
appear: the features may have very different numerical scales.

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

For algorithms that are sensitive to scale, a variable should not dominate
the analysis only because its numerical values are much larger than the
others.

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

With `withMean=True` and `withStd=True`, each feature is standardized using
its mean and standard deviation.

Conceptually:

```text
                       value - mean
standardized value = --------------------
                    standard deviation
```

The purpose is **not** to convert every value to a range between 0 and 1.

Instead, the purpose is to put features with different numerical scales on
a more comparable scale.

For example:

```text
Before scaling

binary feature     → 0 / 1
price              → tens
orders             → hundreds
sales              → thousands

            ↓ StandardScaler

features_scaled
```

This is especially important for algorithms such as PCA and K-Means because
they are sensitive to the scale of the input variables.

The resulting standardized vector is stored in:

```text
features_scaled
```

---

## 8. Saving the Prepared Features

After preparing and standardizing the features, I saved the resulting
DataFrame as a Delta table.

For example:

```python
df_scaled.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("mart_myproject.ml_features")
```

Saving the features was useful because I separated the Machine Learning
process into different steps:

```text
Feature Engineering
        ↓
ml_features
        ↓
PCA
        ↓
K-Means
```

Saving the intermediate dataset is not a requirement for PCA or K-Means.

In my implementation, it was a way to organize the process so that each
step could be developed, executed, and validated separately.

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

---

## What I Learned

The main lesson I learned from this step is that preparing data for Machine
Learning is more than having clean data.

I first need to understand the objective of the analysis, decide which
information can help with that objective, and transform that information
into features that the algorithm can use.

I also learned that:

- Not every column should automatically become a feature.
- Features should be selected or created according to the objective of the analysis.
- An identifier can be useful to create a feature even if the identifier itself is not a useful feature.
- Categorical information may need to be transformed into a numerical representation.
- Missing numerical values can be handled using techniques such as median imputation.
- `VectorAssembler` groups the selected feature columns into one vector.
- `StandardScaler` standardizes features with different numerical scales.
- `StandardScaler` does not mean converting the values to a range between 0 and 1.
- Saving intermediate results can help separate and organize the different stages of a Machine Learning process.

After this process, the data is prepared for the next step of my
implementation: **PCA (Principal Component Analysis)**.

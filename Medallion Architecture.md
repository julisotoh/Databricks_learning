# Medallion Architecture in Databricks

This document explains how I implemented a Medallion Architecture in Databricks using PySpark and Delta Lake.

The purpose of this architecture is to separate the data pipeline into different layers according to the level of transformation and data quality.

The structure I used was:

**Source → Bronze / RAW → Silver / ODS → Gold / MART → Analytics / Machine Learning**

---

## Project Organization

One of the first things I learned was the importance of separating notebooks according to their responsibility.

Instead of keeping connections, ingestion, transformations, and data modeling in the same notebook, I organized the project into different folders.

Example:

```text
project/
│
├── 00_connection
│
├── RAW_Bronze/
│
├── ODS_Silver/
│
└── MART_Gold/
```

The connection configuration was kept in a separate notebook.

From another Databricks notebook, I could execute it using:

```python
%run ../00_notebook_name
```

This helped me avoid repeating connection configuration in multiple notebooks and made the project easier to maintain.

> Note: `%run` executes another Databricks notebook in the current notebook context, making its variables and functions available.

---

# Bronze / RAW Layer

## What I learned

The Bronze or RAW layer is the first layer of the Medallion Architecture.

Its purpose is to preserve the data as close as possible to the original source.

Data can come from different sources such as:

- CSV files
- AWS S3
- Databases
- APIs
- Excel files
- Other external systems

At this stage, I avoid applying business transformations because I want to preserve the original information for traceability.

```text
Source
   │
   ▼
RAW / Bronze
```

## Creating the RAW database

I created a separate database/schema for the RAW layer.

Example:

```python
spark.sql("""
CREATE DATABASE IF NOT EXISTS raw_myproject
""")
```

This keeps the raw tables logically separated from the transformed layers.

## Saving data as Delta tables

After reading the source data into a DataFrame, I persisted it using Delta Lake.

Example:

```python
df_traffic.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("raw_myproject.raw_traffic")
```

Delta Lake allows the data to be stored as managed tables while providing features such as ACID transactions and table history.

---

# Silver / ODS Layer

## What I learned

The Silver or ODS layer contains data that has already gone through cleaning, validation, and standardization.

Instead of reading the original source again, the Silver layer uses the tables stored in the RAW layer.

```text
RAW / Bronze
      │
      ▼
Cleaning
Standardization
Validation
      │
      ▼
ODS / Silver
```

For example:

```python
df_raw = spark.table("raw_myproject.raw_traffic")
```

## Using PySpark functions

For many transformations, I used functions from PySpark SQL:

```python
from pyspark.sql import functions as F
```

Some examples of transformations that can be performed in this layer are:

- Handling null values
- Removing duplicates
- Removing or replacing unwanted characters
- Standardizing categorical values
- Converting data types
- Validating ranges
- Creating audit columns
- Creating identifiers when required
- Tracking the source of each record

For example:

## Adding Audit Columns

One practice I found useful in the Silver / ODS layer was adding audit columns.

Audit columns help identify when a record was processed, when it was updated, where the data came from, and provide an identifier that can be used during the transformation process.

For example:

```python
from pyspark.sql import functions as F

df_silver = (
    df_raw
    .withColumn("created_at", F.current_timestamp())
    .withColumn("updated_at", F.current_timestamp())
    .withColumn("data_source", F.lit("source_system"))
    .withColumn(
        "record_id",
        F.monotonically_increasing_id().cast("string")
    )
)
```

The purpose of each column is:

| Column | Purpose |
|---|---|
| `created_at` | Records when the row was processed or created in this layer |
| `updated_at` | Records when the row was last processed or updated |
| `data_source` | Identifies the system or source from which the data originated |
| `record_id` | Generates an identifier that can be used to distinguish records |

This information is useful for **traceability, troubleshooting, and data lineage**.

For example, if an unexpected record appears later in the pipeline, the audit information can help identify its source and when it was processed.

After applying the transformations, I saved the result in a different database/schema:

```python
spark.sql("""
CREATE DATABASE IF NOT EXISTS ods_myproject
""")
```

```python
df_ods.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("ods_myproject.ods_traffic")
```

Keeping RAW and ODS separated helped me distinguish the original data from the cleaned and standardized version.

---

# Gold / MART Layer

## What I learned

The Gold or MART layer contains data prepared for analytics and business consumption.

For this layer, I implemented a **Star Schema** using dimension and fact tables.

```text
ODS / Silver
      │
      ▼
Dimensions + Facts
      │
      ▼
MART / Gold
      │
      ▼
Analytics / BI / Machine Learning
```

I created another database/schema to keep this layer separated:

```python
spark.sql("""
CREATE DATABASE IF NOT EXISTS mart_myproject
""")
```

---

## Dimension Tables

Dimension tables contain descriptive information that provides context to the business data.

Examples could include:

```text
dim_date
dim_location
dim_vehicle
dim_customer
dim_operation
```

For example, instead of repeating location information in every record of a large fact table, this information can be stored in a dimension.

A dimension may contain:

```text
location_id
city
zone
neighborhood
latitude
longitude
```

The `location_id` can then be referenced from a fact table.

---

## Fact Tables

Fact tables contain the events, observations, or measurements that we want to analyze.

They normally include foreign keys that connect the fact table with the dimensions.

For example:

```text
fact_traffic
--------------------------------
time_id
location_id
operation_id
speed
traffic_intensity
occupancy
```

The relationships can be represented as a Star Schema:

```text
                  dim_time
                      │
                      │
                      ▼
dim_location ─── fact_traffic ─── dim_operation
                      │
                      │
                      ▼
                 dim_vehicle
```

This structure makes it easier to analyze the data using SQL, BI tools, or Machine Learning processes.

---

# Why Separate the Layers?

One of the most useful things I learned while implementing this architecture was that each layer has a different responsibility.

| Layer | Main Purpose |
|---|---|
| **Bronze / RAW** | Preserve data from the source |
| **Silver / ODS** | Clean, validate, and standardize data |
| **Gold / MART** | Prepare data for analytics and business consumption |

This separation also improves traceability.

For example, if I find an unexpected value in a Gold table, I can check the corresponding Silver table to determine whether the issue was introduced during data modeling.

If necessary, I can then compare it with the RAW layer to see how the value originally arrived from the source.

```text
SOURCE
   │
   ▼
RAW / BRONZE
Original data
   │
   ▼
ODS / SILVER
Cleaned and standardized data
   │
   ▼
MART / GOLD
Business-ready data
   │
   ├──► SQL Analytics
   ├──► Tableau / Power BI
   └──► Machine Learning
```

---

# Key Takeaways

By implementing the Medallion Architecture, I learned how to:

- Organize Databricks notebooks by responsibility.
- Reuse common notebook logic with `%run`.
- Separate raw data from transformed data.
- Persist PySpark DataFrames as Delta tables.
- Apply data quality and standardization rules with PySpark.
- Create audit fields for traceability.
- Organize data into RAW, ODS, and MART schemas.
- Build dimension and fact tables.
- Implement a Star Schema for analytical workloads.
- Prepare datasets for BI and Machine Learning.

The most important lesson for me was that the Medallion Architecture is not only about creating Bronze, Silver, and Gold folders or tables.

Each layer has a specific responsibility, which makes the data pipeline easier to understand, troubleshoot, maintain, and extend.

# Medallion Architecture in Databricks — Learning Notes

This document contains my notes and implementation examples while learning how a **Medallion Architecture** can be organized in Databricks using **PySpark and Delta Lake**.

> **Learning note**
>
> This is not intended to represent a complete production architecture.
> It documents the approach I used to understand the responsibilities of the Bronze, Silver, and Gold layers.

---

# What I Wanted to Understand

The main idea I wanted to understand was:

**Why should the same data pass through different layers instead of transforming everything in one step?**

The structure I practiced was:

```text
Data Source
     │
     ▼
Bronze / RAW
     │
     ▼
Silver / ODS
     │
     ▼
 Gold / MART
     │
     ├──────────► Analytics / BI
     │
     └──────────► Machine Learning
```

Each layer has a different responsibility.

| Layer | Main Purpose |
|---|---|
| Bronze / RAW | Preserve source data |
| Silver / ODS | Clean and standardize |
| Gold / MART | Prepare business-ready data |

---

# Project Organization

One of the first things I learned was the importance of separating notebooks according to their responsibility.

Instead of keeping connections, ingestion, transformations, and data modeling in the same notebook, I organized the exercises into different areas.

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

The connection configuration can be kept in a separate notebook.

From another Databricks notebook, it can be executed using:

```python
%run ../00_notebook_name
```

This helped me understand how common configuration can be reused instead of repeating the same code across several notebooks.

> **Note:** `%run` executes another Databricks notebook in the current notebook context, making its variables and functions available.

---

# 1. Bronze / RAW Layer

## What is the Bronze layer?

The Bronze or RAW layer is the first layer of the Medallion Architecture.

Its purpose is to preserve data as close as possible to the original source.

Data may come from:

- CSV files
- AWS S3
- Databases
- APIs
- Excel files
- Other external systems

The idea I followed during this exercise was:

```text
SOURCE
   │
   ▼
BRONZE / RAW
```

At this stage, I avoid applying major business transformations because preserving the original information helps with traceability and troubleshooting.

---

## Creating the RAW schema

Example:

```python
spark.sql("""CREATE DATABASE IF NOT EXISTS raw_myproject""")
```

This keeps the raw tables logically separated from transformed data.

---

## Saving data as Delta tables

After reading the source data into a DataFrame, I persisted it using Delta format.

Example:

```python
df_traffic.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("raw_myproject.raw_traffic")
```

The concept I wanted to practice here was:

```text
Source file
     │
     ▼
Spark DataFrame
     │
     ▼
Delta Table
     │
     ▼
RAW Layer
```

---

## Why keep RAW data?

During this exercise I understood that preserving raw data can be useful for:

- Traceability
- Troubleshooting
- Reprocessing
- Comparing transformed data with its source
- Auditing transformations

This was especially familiar to me because data traceability is also important in traditional ETL architectures.

---

# 2. Silver / ODS Layer

## What is the Silver layer?

The Silver or ODS layer contains data that has already gone through cleaning, validation, or standardization.

Instead of reading the original source again, this layer reads from the RAW tables.

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

---

## Reading from RAW

Example:

```python
df_raw = spark.table("raw_myproject.raw_traffic")
```

At this point I can begin applying transformations.

---

## Example transformations

Some transformations I practiced include:

- Renaming columns
- Casting data types
- Handling null values
- Standardizing descriptions
- Cleaning text
- Removing unnecessary columns
- Creating derived columns
- Adding audit information

Example:

```python
from pyspark.sql import functions as F

df_silver = (
    df_raw
    .withColumn("description", F.trim(F.col("description")))
    .withColumn("processed_at", F.current_timestamp())
)
```

---

## Data quality

One important concept I wanted to understand was that the Silver layer should contain more reliable data than the Bronze layer.

For example, I can inspect null values:

```python
df_silver.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df_silver.columns
]).show()
```

Or inspect duplicate records:

```python
df_silver.groupBy("id") \
    .count() \
    .filter(F.col("count") > 1) \
    .show()
```

These are simple learning examples, but they helped me understand where data-quality controls can be introduced.

---

## Audit columns

I also practiced adding columns that provide information about when data was processed.

Example:

```python
df_silver = (
    df_silver
    .withColumn("created_at", F.current_timestamp())
    .withColumn("source_system", F.lit("source_name"))
)
```

Audit fields can help answer questions such as:

- When was this record processed?
- Where did the data come from?
- Which process generated it?

---

## Saving the Silver table

```python
df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("ods_myproject.ods_traffic")
```

The flow is now:

```text
Source
   │
   ▼
RAW
   │
   ▼
Cleaning / Validation
   │
   ▼
ODS
```

---

# 3. Gold / MART Layer

## What is the Gold layer?

The Gold or MART layer contains data prepared for analytical consumption.

This is where I practiced concepts such as:

- Dimension tables
- Fact tables
- Business-oriented transformations
- Aggregations
- Star Schema modeling

```text
ODS / Silver
      │
      ▼
Business Transformations
      │
      ▼
MART / Gold
      │
      ├────► BI / Dashboards
      │
      └────► Machine Learning
```

---

# Dimension Tables

A dimension contains descriptive information that provides context to facts.

Examples might include:

```text
DIM_DATE
DIM_LOCATION
DIM_VEHICLE
DIM_PROFILE
```

A simplified dimension can be created using PySpark.

Example:

```python
dim_location = (
    df_silver
    .select(
        "location_id",
        "location_name"
    )
    .dropDuplicates()
)
```

Then persisted as a Delta table:

```python
dim_location.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("mart_myproject.dim_location")
```

---

# Fact Tables

A fact table normally contains measurable events and references to dimensions.

Example conceptual structure:

```text
FACT_EVENT
│
├── date_id
├── location_id
├── vehicle_id
├── event_count
└── measurement
```

A simplified example:

```python
fact_event = df_silver.select(
    "event_id",
    "date_id",
    "location_id",
    "vehicle_id",
    "measurement"
)
```

Save it:

```python
fact_event.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("mart_myproject.fact_event")
```

---

# Star Schema

This exercise also helped me understand how dimensions and facts can form a Star Schema.

```text
                  DIM_DATE
                     │
                     │
DIM_LOCATION ─── FACT_EVENT ─── DIM_VEHICLE
                     │
                     │
                 DIM_PROFILE
```

The fact table contains the events or measurements.

The dimensions provide context for analyzing those events.

---

# Why Separate the Layers?

Before working with this architecture, one question I had was:

> Why not clean and transform everything directly when reading the source?

The separation helped me understand several benefits.

## Traceability

If a transformation produces an unexpected result, I can compare:

```text
SOURCE
  ↓
RAW
  ↓
ODS
  ↓
MART
```

and identify at which stage the data changed.

---

## Reprocessing

Because RAW preserves source information, transformations can potentially be executed again without having to obtain the source file again.

---

## Separation of responsibilities

Each layer has a specific purpose:

```text
RAW  = What arrived?

ODS  = What does the cleaned data look like?

MART = How will the data be consumed?
```

This made the pipeline easier for me to understand and troubleshoot.

---

# Delta Lake

Another concept I practiced during this implementation was **Delta Lake**.

Instead of working only with temporary Spark DataFrames, I persisted data between stages as Delta tables.

Simplified example:

```python
df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("schema.table")
```

This allowed me to work with persistent tables between notebooks and pipeline stages.

---

# Full Learning Flow

Putting the concepts together:

```text
                 DATA SOURCE
                     │
                     ▼
                   AWS S3
                     │
                     ▼
             ┌────────────────┐
             │ BRONZE / RAW   │
             │ Original data  │
             └───────┬────────┘
                     │
                     ▼
             ┌────────────────┐
             │ SILVER / ODS   │
             │ Cleaned data   │
             └───────┬────────┘
                     │
                     ▼
             ┌────────────────┐
             │ GOLD / MART    │
             │ Business data  │
             └───────┬────────┘
                     │
             ┌───────┴────────┐
             ▼                ▼
       Analytics / BI    Machine Learning
```

---


# Important Note

This repository documents my **learning process**.

The examples are intentionally simplified so I can focus on understanding individual concepts.

---

## Next Step

After preparing the data, I started exploring how the Gold/MART information could be prepared for Machine Learning.

➡️ [Machine Learning — Feature Engineering](Machine_Learning_feature.md)

---

⬅️ [Back to main README](README.md)

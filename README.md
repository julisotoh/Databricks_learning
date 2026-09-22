# Databricks Learning Journey

This repository documents part of my learning journey with **Databricks, PySpark, Delta Lake, AWS, and Machine Learning**.

I created it while studying for my Master's in Big Data & Artificial Intelligence and while practicing how different data engineering and machine learning concepts can be implemented in Databricks.

The purpose of this repository is **not to present myself as a Databricks expert**, but to document what I am learning through hands-on practice and to create notes and examples that may also be useful for other people learning these technologies.

> **Learning repository**
>
> The examples in this repository reflect my learning process.  
> They may evolve as I continue studying, testing new approaches, and improving my understanding of Databricks and the modern data stack.

---

## What I am learning

Through these exercises I have been practicing:

- Databricks
- PySpark
- Delta Lake
- AWS S3 integration
- Databricks CLI
- Databricks Secret Scopes
- Medallion Architecture
- Data transformation and data quality
- Dimensional modeling
- Machine Learning with Spark ML
- Feature engineering
- Feature scaling
- PCA (Principal Component Analysis)
- K-Means clustering

---

## Learning Architecture

One of the main exercises documented in this repository follows this general flow:

```text
Data Sources
     │
     ▼
   AWS S3
     │
     ▼
 Databricks
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
     ├────────────► Analytics / BI
     │
     └────────────► Machine Learning
                        │
                        ▼
                  Feature Engineering
                        │
                        ▼
                       PCA
                        │
                        ▼
                     K-Means

# AWS S3 + Databricks — Learning Notes

This document contains my notes from practicing how to connect **AWS S3 with Databricks**.

The objective of this exercise was to understand how Databricks can access files stored in AWS while avoiding the practice of writing credentials directly inside notebooks.

> **Learning note**
>
> This configuration reflects the approach I used during my learning process.
> It is intended as a practical example rather than a production security architecture.

---

## What I Wanted to Understand

Before doing this exercise, I wanted to understand how these components interact:

```text
AWS S3
   │
   ▼
AWS IAM
   │
   ▼
Databricks Secret Scope
   │
   ▼
PySpark
   │
   ▼
Databricks DataFrame
```

The main concepts I practiced were:

- Databricks CLI
- Databricks Access Tokens
- AWS IAM credentials
- Databricks Secret Scopes
- `dbutils.secrets`
- Reading files from S3 using PySpark

---

# 1. Install Databricks CLI on Mac

From the terminal:

```bash
pip3 install databricks-cli
```

The CLI allows Databricks resources to be managed from the terminal.

---

# 2. Create a Databricks Access Token

From the Databricks Workspace:

```text
Settings
   ↓
Developer
   ↓
Access Tokens
   ↓
Manage
```

Create a token and configure:

- Name
- Expiration period
- Required permissions

Then configure Databricks CLI:

```bash
databricks configure --token
```

Enter:

```text
Databricks Host
Access Token
```

> Access tokens should be treated as credentials and should never be committed to GitHub.

---

# 3. Create a Secret Scope

First, check the existing Secret Scopes:

```bash
databricks secrets list-scopes
```

Create a new one if needed:

```bash
databricks secrets create-scope --scope <scope_name>
```

A Secret Scope provides a way to reference sensitive values without writing them directly inside a notebook.

---

# 4. Configure AWS IAM

For this learning exercise, I created an IAM user with permissions to access the S3 bucket used by the project.

Instead of writing the AWS credentials directly in the notebooks, I stored them in a Databricks Secret Scope.

Store the Access Key:

```bash
databricks secrets put \
  --scope <scope_name> \
  --key aws-access-key \
  --string-value "ACCESS_KEY_ID"
```

Store the Secret Access Key:

```bash
databricks secrets put \
  --scope <scope_name> \
  --key aws-secret-key \
  --string-value "SECRET_ACCESS_KEY"
```

Verify the stored keys:

```bash
databricks secrets list --scope <scope_name>
```

> The values shown here are placeholders.
>
> Real AWS credentials should never be stored in source code or committed to a public repository.

---

# 5. Access Secrets from Databricks

From a Databricks notebook, I retrieved the stored credentials using `dbutils.secrets.get()`:

```python
access_key = dbutils.secrets.get(
    scope="<scope_name>",
    key="aws-access-key"
)

secret_key = dbutils.secrets.get(
    scope="<scope_name>",
    key="aws-secret-key"
)
```

This allowed me to use the credentials without writing their actual values directly in the notebook.

---

# 6. Read CSV Files from AWS S3 with PySpark

Once the credentials were available, I could read CSV files stored in S3:

```python
dataframe = (
    spark.read
    .option("fs.s3a.access.key", access_key)
    .option("fs.s3a.secret.key", secret_key)
    .option("fs.s3a.endpoint", "s3.amazonaws.com")
    .option("header", "true")
    .option("delimiter", ";")
    .option("inferSchema", "true")
    .option("encoding", "UTF-8")
    .csv("<s3_csv_file_path>")
)
```

At this point, the S3 file becomes a Spark DataFrame that can be processed in Databricks.

---

# Data Flow

This exercise helped me understand the following flow:

```text
CSV File
   │
   ▼
AWS S3
   │
   ▼
AWS IAM
   │
   ▼
Databricks Secret Scope
   │
   ▼
PySpark
   │
   ▼
Spark DataFrame
```

From this point, the data can begin moving through the Medallion Architecture:

```text
Source
   │
   ▼
Bronze / RAW
   │
   ▼
Silver / ODS
   │
   ▼
Gold / MART
```

The next step is documented here:

 [Medallion Architecture](Medallion_Architecture.md)

---


# Important Note

This is a **learning implementation**.

For production environments, authentication and cloud access should follow the security architecture and governance standards defined by the organization.

The purpose of this example is to document the concepts I practiced while learning how AWS and Databricks can work together.

---

[Back to main README](README.md)

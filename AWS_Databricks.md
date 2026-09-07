#  AWS S3 + Databricks Setup

This guide documents the configuration I used to connect **AWS S3 to Databricks from my Mac**, using Databricks CLI, AWS IAM, and Databricks Secret Scopes.


---

## 1. Install Databricks CLI on Mac

From the terminal:

```bash
pip3 install databricks-cli
```

---

## 2. Create a Databricks Access Token

From your Databricks Workspace:

**Settings → Developer → Access Tokens → Manage**

Create a token and configure:

- Name
- Expiration period
- Required permissions

Then configure Databricks CLI:

```bash
databricks configure --token
```

Enter:

- Databricks Host
- Access Token

---

## 3. Create a Secret Scope

Check the existing Secret Scopes:

```bash
databricks secrets list-scopes
```

Create a new one if needed:

```bash
databricks secrets create-scope --scope <scope_name>
```

---

## 4. Configure AWS IAM

In AWS, I created an **IAM user** with permissions to access the S3 bucket used in the project.

Instead of writing the AWS credentials directly in the notebooks, I stored them in a Databricks Secret Scope.

Store the Access Key:

```bash
databricks secrets put --scope <nombre_scope> --key aws-access-key --string-value "ACCESS_KEY_ID"
```

Store the Secret Access Key:

```bash
databricks secrets put --scope <nombre_scope> --key aws-secret-key --string-value "SECRET_ACCESS_KEY"
```

Verify the stored keys:

```bash
databricks secrets list --scope <scope_name>
```

---

## 5. Access the Secrets from Databricks

From a Databricks notebook, I used `dbutils.secrets.get()` to retrieve the stored credentials:

```python
access_key = dbutils.secrets.get(scope="<scope_name>", key="aws-access-key")

secret_key = dbutils.secrets.get( scope="<scope_name>",  key="aws-secret-key")
```

This way, the credentials are not written directly in the notebook code.

---

## 6. Read CSV Files from AWS S3 with PySpark

Once the credentials are retrieved from the Secret Scope, the CSV files stored in S3 can be read from Databricks:

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

---

## Data Flow

**CSV → AWS S3 → IAM → Secret Scope → PySpark → Databricks**

From this point, the data can begin moving through the Medallion Architecture:

**Bronze → Silver → Gold**

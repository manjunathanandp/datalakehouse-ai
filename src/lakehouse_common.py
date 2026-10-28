import os

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.io.pyarrow import schema_to_pyarrow
from pyiceberg.schema import Schema
from pyiceberg.types import DoubleType, LongType, NestedField, StringType


BRONZE_SCHEMA = Schema(
    NestedField(1, "transaction_id", LongType(), required=True),
    NestedField(2, "customer_id", StringType(), required=True),
    NestedField(3, "customer_name", StringType(), required=True),
    NestedField(4, "parent_company", StringType(), required=True),
    NestedField(5, "amount", DoubleType(), required=True),
    NestedField(6, "status", StringType(), required=True),
    NestedField(7, "transaction_date", StringType(), required=True),
    NestedField(8, "region", StringType(), required=True),
)

SILVER_SCHEMA = Schema(
    NestedField(1, "transaction_id", LongType(), required=True),
    NestedField(2, "customer_id", StringType(), required=True),
    NestedField(3, "customer_name", StringType(), required=True),
    NestedField(4, "parent_company", StringType(), required=True),
    NestedField(5, "amount", DoubleType(), required=True),
    NestedField(6, "status", StringType(), required=True),
    NestedField(7, "transaction_date", StringType(), required=True),
    NestedField(8, "region", StringType(), required=True),
    NestedField(9, "risk_level", StringType(), required=True),
)

GOLD_SCHEMA = Schema(
    NestedField(1, "customer_id", StringType(), required=True),
    NestedField(2, "customer_name", StringType(), required=True),
    NestedField(3, "parent_company", StringType(), required=True),
    NestedField(4, "risk_level", StringType(), required=True),
    NestedField(5, "total_spend", DoubleType(), required=True),
    NestedField(6, "flagged_volume", DoubleType(), required=True),
    NestedField(7, "transaction_count", LongType(), required=True),
    NestedField(8, "average_amount", DoubleType(), required=True),
)

TABLE_SCHEMAS = {
    ("demo_bronze", "transactions"): BRONZE_SCHEMA,
    ("demo_silver", "transactions"): SILVER_SCHEMA,
    ("demo_gold", "customer_metrics"): GOLD_SCHEMA,
}


def load_lakehouse_catalog() -> Catalog:
    return load_catalog(
        "local",
        uri=os.getenv("ICEBERG_REST_URI", "http://localhost:8182"),
        **{
            "s3.endpoint": os.getenv("S3_ENDPOINT", "http://localhost:9000"),
            "s3.access-key-id": os.getenv("S3_ACCESS_KEY_ID", "admin"),
            "s3.secret-access-key": os.getenv("S3_SECRET_ACCESS_KEY", "password123"),
            "s3.region": os.getenv("AWS_REGION", "us-east-1"),
            "s3.path-style-access": "true",
        },
    )


def write_iceberg_table(
    catalog: Catalog, identifier: tuple[str, str], frame: pd.DataFrame
) -> None:
    schema = TABLE_SCHEMAS[identifier]
    namespace = identifier[0]
    if (namespace,) not in catalog.list_namespaces():
        catalog.create_namespace(namespace)

    table = catalog.create_table_if_not_exists(
        identifier, schema=schema, properties={"lakehouse.demo": "true"}
    )
    if table.properties.get("lakehouse.demo") != "true":
        raise ValueError(
            f"Refusing to overwrite non-demo Iceberg table {'.'.join(identifier)}."
        )
    arrow_schema = schema_to_pyarrow(schema, include_field_ids=False)
    arrow_table = pa.Table.from_pandas(
        frame, schema=arrow_schema, preserve_index=False
    )
    table.overwrite(arrow_table)


def read_iceberg_table(catalog: Catalog, identifier: tuple[str, str]) -> pd.DataFrame:
    return catalog.load_table(identifier).scan().to_arrow().to_pandas()

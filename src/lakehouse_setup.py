import csv
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from lakehouse_common import (
    GOLD_SCHEMA,
    SILVER_SCHEMA,
    load_lakehouse_catalog,
    write_iceberg_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = PROJECT_ROOT / "data" / "source" / "transactions.csv"
SOURCE_FIELDS = [
    "transaction_id",
    "customer_id",
    "customer_name",
    "parent_company",
    "amount",
    "status",
    "transaction_date",
    "region",
]
CUSTOMERS = [
    ("CUST_01", "Northwind Traders", "Northwind Group", "West"),
    ("CUST_02", "Contoso Retail", "Contoso Group", "East"),
    ("CUST_03", "Fabrikam Services", "Fabrikam Group", "Central"),
    ("CUST_04", "Adventure Works", "Adventure Group", "West"),
    ("CUST_05", "Wide World Importers", "World Imports", "East"),
    ("CUST_06", "Tailspin Toys", "Tailspin Group", "Central"),
    ("CUST_99", "Alpha Corp", "MegaConglomerate Inc", "North"),
]


def generate_demo_source() -> None:
    SOURCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(42)
    first_day = date(2026, 1, 1)
    regular_statuses = [
        "SETTLED", "SETTLED", "SETTLED", "PENDING", "SETTLED", "FLAGGED",
        "SETTLED", "SETTLED", "PENDING", "SETTLED", "SETTLED", "SETTLED",
    ]
    with SOURCE_FILE.open("w", newline="", encoding="utf-8") as source:
        writer = csv.DictWriter(source, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        transaction_id = 1
        for customer_id, customer_name, parent, region in CUSTOMERS:
            for index in range(12):
                status = (
                    "FLAGGED" if customer_id == "CUST_99" and index % 2 == 0
                    else regular_statuses[index]
                )
                writer.writerow(
                    {
                        "transaction_id": transaction_id,
                        "customer_id": customer_id,
                        "customer_name": customer_name,
                        "parent_company": parent,
                        "amount": round(randomizer.uniform(25, 5000), 2),
                        "status": status,
                        "transaction_date": (
                            first_day + timedelta(days=randomizer.randrange(270))
                        ).isoformat(),
                        "region": region,
                    }
                )
                transaction_id += 1
    print(f"Generated reproducible synthetic source data: {SOURCE_FILE}")


def _build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    cleaned = bronze.copy()
    cleaned["transaction_id"] = pd.to_numeric(
        cleaned["transaction_id"], errors="raise"
    ).astype("int64")
    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="raise").astype(float)
    if cleaned[SOURCE_FIELDS].isna().any().any():
        raise ValueError("Source transactions contain required empty fields.")
    for column in ("customer_id", "customer_name", "parent_company", "status", "region"):
        cleaned[column] = cleaned[column].astype(str).str.strip()
    cleaned["status"] = cleaned["status"].str.upper()
    cleaned["transaction_date"] = pd.to_datetime(
        cleaned["transaction_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    if cleaned[
        ["customer_id", "customer_name", "parent_company", "status", "region"]
    ].eq("").any().any():
        raise ValueError("Source transactions contain required empty fields.")

    cleaned = cleaned.drop_duplicates(subset=["transaction_id"], keep="first")
    cleaned = cleaned.loc[
        cleaned["customer_id"].ne("")
        & cleaned["customer_name"].ne("")
        & cleaned["parent_company"].ne("")
        & cleaned["amount"].ge(0)
        & cleaned["status"].isin({"SETTLED", "PENDING", "FLAGGED"})
    ].copy()
    if cleaned.empty:
        raise ValueError("No valid transactions remain after Silver-layer checks.")

    flagged_share = cleaned.assign(
        flagged=cleaned["status"].eq("FLAGGED")
    ).groupby("customer_id")["flagged"].mean()
    risk_levels = flagged_share.ge(0.5).map({True: "HIGH", False: "STANDARD"})
    cleaned["risk_level"] = cleaned["customer_id"].map(risk_levels)
    return cleaned[SILVER_SCHEMA.column_names]


def _build_gold(silver: pd.DataFrame) -> pd.DataFrame:
    metrics = silver.copy()
    metrics["flagged_amount"] = metrics["amount"].where(
        metrics["status"].eq("FLAGGED"), 0.0
    )
    gold = (
        metrics.groupby(
            ["customer_id", "customer_name", "parent_company", "risk_level"],
            as_index=False,
        )
        .agg(
            total_spend=("amount", "sum"),
            flagged_volume=("flagged_amount", "sum"),
            transaction_count=("transaction_id", "count"),
            average_amount=("amount", "mean"),
        )
        .round({"total_spend": 2, "flagged_volume": 2, "average_amount": 2})
    )
    gold["transaction_count"] = gold["transaction_count"].astype("int64")
    return gold[GOLD_SCHEMA.column_names]


def run_pipeline() -> None:
    if not SOURCE_FILE.exists():
        generate_demo_source()
    bronze = pd.read_csv(SOURCE_FILE)
    missing_fields = sorted(set(SOURCE_FIELDS) - set(bronze.columns))
    if missing_fields:
        raise ValueError(f"Source CSV is missing columns: {', '.join(missing_fields)}")
    bronze = bronze[SOURCE_FIELDS].copy()
    bronze["transaction_id"] = pd.to_numeric(
        bronze["transaction_id"], errors="raise"
    ).astype("int64")
    bronze["amount"] = pd.to_numeric(bronze["amount"], errors="raise").astype(float)

    silver = _build_silver(bronze)
    gold = _build_gold(silver)
    catalog = load_lakehouse_catalog()
    write_iceberg_table(catalog, ("demo_bronze", "transactions"), bronze)
    write_iceberg_table(catalog, ("demo_silver", "transactions"), silver)
    write_iceberg_table(catalog, ("demo_gold", "customer_metrics"), gold)
    print(
        "Iceberg medallion pipeline completed: "
        f"Bronze={len(bronze)}, Silver={len(silver)}, Gold={len(gold)}"
    )


if __name__ == "__main__":
    run_pipeline()

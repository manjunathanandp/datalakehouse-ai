import argparse
import json
import os
import re
from pathlib import Path
from datetime import date
from typing import Any, TypedDict

import pandas as pd
import requests
from neo4j import GraphDatabase

from lakehouse_common import load_lakehouse_catalog, read_iceberg_table


SEMANTIC_MODEL_FILE = Path(__file__).with_name("semantic_model.json")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OPA_URL = os.getenv("OPA_URL", "http://localhost:8181/v1/data/lakehouse/authz/allow")


class QueryIntent(TypedDict):
    measure: str
    customer_id: str | None
    transaction_date: str | None
    status: str | None
    operation: str


def _validate_query_intent(intent: Any, semantic_model: dict[str, Any]) -> QueryIntent:
    if not isinstance(intent, dict):
        raise ValueError("Ollama query intent must be a JSON object.")
    valid_measures = {metric["name"] for metric in semantic_model["measures"]}
    measure = intent.get("measure")
    customer_id = intent.get("customer_id")
    transaction_date = intent.get("transaction_date")
    status = intent.get("status")
    operation = intent.get("operation")

    if not isinstance(measure, str) or measure not in valid_measures:
        raise ValueError(f"Ollama selected an unknown semantic measure: {measure!r}.")
    if customer_id is not None and (
        not isinstance(customer_id, str) or not customer_id.strip()
    ):
        raise ValueError("Ollama returned an invalid customer_id.")
    if transaction_date is not None:
        if not isinstance(transaction_date, str):
            raise ValueError("Ollama returned an invalid transaction_date.")
        try:
            parsed_date = date.fromisoformat(transaction_date)
        except ValueError as error:
            raise ValueError(
                f"Ollama returned an invalid transaction_date: {transaction_date!r}."
            ) from error
        if parsed_date.isoformat() != transaction_date:
            raise ValueError("transaction_date must use YYYY-MM-DD format.")
    if status is not None and (
        not isinstance(status, str)
        or status.upper() not in {"SETTLED", "PENDING", "FLAGGED"}
    ):
        raise ValueError(f"Ollama returned an unsupported transaction status: {status!r}.")
    if operation not in {"aggregate", "top_customer"}:
        raise ValueError(f"Ollama returned an unsupported operation: {operation!r}.")

    return {
        "measure": measure,
        "customer_id": customer_id,
        "transaction_date": transaction_date,
        "status": status.upper() if status else None,
        "operation": operation,
    }


def load_semantic_model() -> dict[str, Any]:
    with SEMANTIC_MODEL_FILE.open(encoding="utf-8") as model_file:
        return json.load(model_file)


def ollama_chat(messages: list[dict[str, str]], json_mode: bool = False) -> str:
    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120
        )
        response.raise_for_status()
    except requests.HTTPError as error:
        response_body = error.response.text[:500] if error.response is not None else ""
        detail = f" HTTP response: {response_body}" if response_body else ""
        raise RuntimeError(
            f"Ollama returned an HTTP error at {OLLAMA_BASE_URL}: "
            f"{error}.{detail}"
        ) from error
    except requests.RequestException as error:
        raise RuntimeError(
            f"Ollama request to {OLLAMA_BASE_URL} failed "
            f"({type(error).__name__}: {error}). Verify that Ollama is running, "
            "reachable from this Python process, and configured with the expected "
            "OLLAMA_BASE_URL."
        ) from error

    try:
        response_data = response.json()
    except requests.JSONDecodeError as error:
        raise ValueError("Ollama returned a response that was not valid JSON.") from error
    if not isinstance(response_data, dict):
        raise ValueError("Ollama returned an invalid chat response.")
    content = response_data.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned an empty or invalid chat response.")
    return content.strip()


def extract_query_intent(
    question: str, semantic_model: dict[str, Any]
) -> QueryIntent:
    question_text = question.casefold()
    normalized_question = re.sub(r"[^a-z0-9]+", " ", question_text).strip()
    matched_measure: str | None = None
    matched_alias = ""
    for metric in semantic_model["measures"]:
        aliases = [metric["name"], *metric["synonyms"]]
        for alias in aliases:
            normalized_alias = re.sub(r"[^a-z0-9]+", " ", alias.casefold()).strip()
            if normalized_alias and re.search(
                rf"\b{re.escape(normalized_alias)}\b", normalized_question
            ):
                if len(normalized_alias) > len(matched_alias):
                    matched_measure = metric["name"]
                    matched_alias = normalized_alias

    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", question_text)
    transaction_date = date_match.group(0) if date_match else None
    if transaction_date:
        try:
            parsed_date = date.fromisoformat(transaction_date)
        except ValueError as error:
            raise ValueError(
                f"Question contains an invalid date: {transaction_date!r}."
            ) from error
        if parsed_date.isoformat() != transaction_date:
            raise ValueError("Dates in questions must use YYYY-MM-DD format.")

    status_match = re.search(r"\b(settled|pending|flagged)\b", question_text)
    status = status_match.group(1).upper() if status_match else None
    customer_match = re.search(r"\bCUST[_ -]?(\d+)\b", question, re.IGNORECASE)
    customer_id = (
        f"CUST_{customer_match.group(1).zfill(2)}" if customer_match else None
    )
    operation = (
        "top_customer"
        if re.search(
            r"\bwhich\s+customers?\b|\bwho\s+(?:had|has|have)\b|"
            r"\b(?:highest|higest|largest|most)\b",
            question_text,
        )
        else "aggregate"
    )

    if matched_measure and operation == "top_customer" and transaction_date:
        return _validate_query_intent(
            {
                "measure": matched_measure,
                "customer_id": customer_id,
                "transaction_date": transaction_date,
                "status": status,
                "operation": operation,
            },
            semantic_model,
        )

    metric_catalog = [
        {
            "name": metric["name"],
            "description": metric["description"],
            "synonyms": metric["synonyms"],
            "dataset": metric["dataset"],
            "filters": metric.get("filters", {}),
        }
        for metric in semantic_model["measures"]
    ]
    content = ollama_chat(
        [
            {
                "role": "system",
                "content": (
                    "Translate the user's analytics question into a query intent "
                    "using only the supplied semantic metric catalog. For questions "
                    "about pending amounts, use pending_amount. Extract an explicit "
                    "date as YYYY-MM-DD and an explicit status as SETTLED, PENDING, "
                    "or FLAGGED. Use top_customer when asked which customer has the "
                    "highest/most value; otherwise use aggregate. Extract a customer "
                    "ID only when explicitly named. Return JSON with exactly these "
                    'keys: {"measure":"catalog metric name","customer_id":"ID or '
                    'null","transaction_date":"YYYY-MM-DD or null","status":"'
                    'SETTLED|PENDING|FLAGGED or null","operation":"aggregate|'
                    'top_customer"}. Apply catalog-defined filters, do not invent '
                    "metrics or filters, and never generate SQL."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "measures": metric_catalog}
                ),
            },
        ],
        json_mode=True,
    )
    try:
        intent = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Ollama did not return valid JSON for the query intent.") from error

    return _validate_query_intent(intent, semantic_model)


def get_graph_context(customer_id: str | None) -> list[dict[str, str]]:
    if customer_id:
        query = """
        MATCH (c:Customer {id: $customer_id})-[:SUBSIDIARY_OF]->(p:ParentCompany)
        MATCH (c)-[:CATEGORIZED_AS]->(r:RiskProfile)
        RETURN c.id AS customer_id, c.name AS customer_name,
               p.name AS parent_company, r.level AS risk_level
        """
        parameters = {"customer_id": customer_id}
    else:
        query = """
        MATCH (c:Customer)-[:SUBSIDIARY_OF]->(p:ParentCompany)
        MATCH (c)-[:CATEGORIZED_AS]->(r:RiskProfile)
        RETURN c.id AS customer_id, c.name AS customer_name,
               p.name AS parent_company, r.level AS risk_level
        ORDER BY c.id
        """
        parameters = {}

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password123"),
        ),
    )
    try:
        with driver.session() as session:
            return [dict(record) for record in session.run(query, parameters)]
    finally:
        driver.close()


def check_opa_governance(agent_role: str, classification: str) -> bool:
    payload = {
        "input": {
            "action": "READ",
            "principal": {"role": agent_role},
            "resource": {"classification": classification},
        }
    }
    try:
        response = requests.post(OPA_URL, json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(f"OPA authorization request failed at {OPA_URL}.") from error

    result = response.json()
    if not isinstance(result, dict) or not isinstance(result.get("result"), bool):
        raise ValueError("OPA response did not contain a boolean authorization result.")
    return result["result"]


def calculate_measure(
    gold: pd.DataFrame, measure_name: str, customer_id: str | None
) -> float | int:
    semantic_model = load_semantic_model()
    definitions = {item["name"]: item for item in semantic_model["measures"]}
    if measure_name not in definitions:
        raise ValueError(f"Unknown semantic measure: {measure_name}")

    selected = gold if customer_id is None else gold.loc[gold["customer_id"].eq(customer_id)]
    if selected.empty:
        target = f" for customer {customer_id}" if customer_id else ""
        raise ValueError(f"No Gold-layer customer metrics were found{target}.")

    definition = definitions[measure_name]
    if definition["aggregation"] == "weighted_average":
        total = float(selected["total_spend"].sum())
        count = int(selected["transaction_count"].sum())
        return round(total / count, 2) if count else 0.0

    column = definition["physical_column"]
    value = selected[column].sum()
    return int(value) if measure_name == "transaction_count" else round(float(value), 2)


def calculate_top_customer(
    silver: pd.DataFrame,
    measure_name: str,
    transaction_date: str | None,
    status: str | None,
    customer_id: str | None = None,
) -> dict[str, Any] | None:
    semantic_model = load_semantic_model()
    definitions = {item["name"]: item for item in semantic_model["measures"]}
    if measure_name not in definitions:
        raise ValueError(f"Unknown semantic measure: {measure_name}")

    definition = definitions[measure_name]
    if definition["dataset"] != "demo_silver.transactions":
        raise ValueError(
            f"Measure {measure_name!r} is not available at transaction-level grain."
        )

    selected = silver.copy()
    for column, value in definition.get("filters", {}).items():
        selected = selected.loc[selected[column].eq(value)]
    if status:
        selected = selected.loc[selected["status"].eq(status)]
    if transaction_date:
        selected = selected.loc[
            selected["transaction_date"].astype(str).eq(transaction_date)
        ]
    if customer_id:
        selected = selected.loc[selected["customer_id"].eq(customer_id)]
    if selected.empty:
        return None

    amount_column = definition["physical_column"]
    ranked = (
        selected.groupby(["customer_id", "customer_name"], as_index=False)
        .agg(amount=(amount_column, "sum"), transaction_count=(amount_column, "size"))
        .sort_values(["amount", "customer_id"], ascending=[False, True])
    )
    winner = ranked.iloc[0]
    return {
        "customer_id": str(winner["customer_id"]),
        "customer_name": str(winner["customer_name"]),
        "amount": round(float(winner["amount"]), 2),
        "transaction_count": int(winner["transaction_count"]),
        "transaction_date": transaction_date,
        "status": status or definition.get("filters", {}).get("status"),
    }


def answer_question(
    question: str,
    role: str = "analytics_agent",
    measure: str | None = None,
    customer_id: str | None = None,
) -> bool:
    semantic_model = load_semantic_model()
    if measure is None:
        intent = extract_query_intent(question, semantic_model)
        measure = intent["measure"]
        customer_id = intent["customer_id"]
        transaction_date = intent["transaction_date"]
        status = intent["status"]
        operation = intent["operation"]
    elif measure not in {item["name"] for item in semantic_model["measures"]}:
        raise ValueError(f"Unknown semantic measure: {measure}")
    else:
        transaction_date = None
        status = None
        operation = "aggregate"

    graph_context = get_graph_context(customer_id)
    if customer_id and not graph_context:
        raise ValueError(f"Customer {customer_id!r} was not found in the knowledge graph.")
    if customer_id is None and not graph_context:
        raise ValueError("The knowledge graph has no customer context; run the demo pipeline.")

    classification = (
        "PII"
        if any(entity["risk_level"] == "HIGH" for entity in graph_context)
        else "PUBLIC"
    )
    if not check_opa_governance(role, classification):
        print(f"Access denied by OPA: role={role}, classification={classification}.")
        return False

    catalog = load_lakehouse_catalog()
    measure_description = next(
        item["description"]
        for item in semantic_model["measures"]
        if item["name"] == measure
    )
    definition = next(
        item for item in semantic_model["measures"] if item["name"] == measure
    )
    if operation == "top_customer":
        if definition["dataset"] != "demo_silver.transactions":
            raise ValueError(
                f"Top-customer ranking is not supported for {measure!r}; "
                "select a transaction-level semantic measure."
            )
        silver = read_iceberg_table(catalog, ("demo_silver", "transactions"))
        result = calculate_top_customer(
            silver, measure, transaction_date, status, customer_id
        )
        if result is None:
            print(
                "No matching transactions found for "
                f"measure={measure}, date={transaction_date}, status={status}."
            )
            return True
        winner_context = get_graph_context(result["customer_id"])
        if not winner_context:
            raise ValueError(
                f"Top customer {result['customer_id']!r} was not found in the knowledge graph."
            )
        graph_context = winner_context
    else:
        if definition["dataset"] != "demo_gold.customer_metrics":
            raise ValueError(
                f"Aggregate queries for transaction-level measure {measure!r} "
                "must use the top-customer operation."
            )
        gold = read_iceberg_table(catalog, ("demo_gold", "customer_metrics"))
        result = calculate_measure(gold, measure, customer_id)
    evidence = {
        "question": question,
        "semantic_measure": measure,
        "measure_definition": measure_description,
        "result": result,
        "customer_filter": customer_id,
        "transaction_date": transaction_date,
        "transaction_status": status,
        "operation": operation,
        "graph_context": graph_context,
        "classification": classification,
    }
    if operation == "top_customer":
        entity = graph_context[0]
        answer = (
            f"{result['customer_name']} ({result['customer_id']}) had the highest "
            f"{result['status'].lower()} amount on {result['transaction_date']}: "
            f"{result['amount']:.2f} across {result['transaction_count']} "
            f"transaction(s). Graph context: this customer is a subsidiary of "
            f"{entity['parent_company']} and has risk level {entity['risk_level']}."
        )
    else:
        answer = ollama_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using only the supplied evidence. "
                        "Report the computed semantic metric and relevant graph "
                        "context. Do not invent facts or recalculate the result."
                    ),
                },
                {"role": "user", "content": json.dumps(evidence)},
            ]
        )
    if operation == "top_customer":
        print(f"\nTop customer result for {measure}: {json.dumps(result)}")
    else:
        print(f"\nSemantic metric: {measure} = {result}")
    print(f"Graph context: {json.dumps(graph_context)}")
    print(f"OPA: authorized role={role} for {classification}")
    print(f"\n{answer}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query semantic Iceberg metrics with Neo4j Graph RAG and Ollama."
    )
    parser.add_argument("--question", help="Natural-language analytics question.")
    parser.add_argument("--role", default="analytics_agent")
    parser.add_argument("--measure", help="Optional explicit semantic metric.")
    parser.add_argument("--customer-id", help="Optional explicit customer filter.")
    args = parser.parse_args()

    if args.question:
        answer_question(args.question, args.role, args.measure, args.customer_id)
        return 0
    if args.measure or args.customer_id:
        parser.error("--measure and --customer-id require --question.")

    print("Lakehouse Graph RAG demo. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if question:
            answer_question(question, args.role)


if __name__ == "__main__":
    raise SystemExit(main())

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lakehouse_setup import _build_gold, _build_silver
from rag_orchestrator import (
    calculate_measure,
    calculate_top_customer,
    extract_query_intent,
    load_semantic_model,
    ollama_chat,
)


class MedallionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = pd.DataFrame(
            [
                {
                    "transaction_id": 1,
                    "customer_id": "CUST_99",
                    "customer_name": "Alpha Corp",
                    "parent_company": "MegaConglomerate Inc",
                    "amount": 100.0,
                    "status": "flagged",
                    "transaction_date": "2026-01-01",
                    "region": "North",
                },
                {
                    "transaction_id": 2,
                    "customer_id": "CUST_99",
                    "customer_name": "Alpha Corp",
                    "parent_company": "MegaConglomerate Inc",
                    "amount": 300.0,
                    "status": "settled",
                    "transaction_date": "2026-01-02",
                    "region": "North",
                },
            ]
        )

    def test_silver_normalizes_and_classifies_flagged_customers(self) -> None:
        silver = _build_silver(self.source)

        self.assertEqual(silver["status"].tolist(), ["FLAGGED", "SETTLED"])
        self.assertEqual(silver["risk_level"].tolist(), ["HIGH", "HIGH"])

    def test_silver_rejects_missing_required_values(self) -> None:
        source = self.source.copy()
        source.loc[0, "parent_company"] = None

        with self.assertRaisesRegex(ValueError, "required empty fields"):
            _build_silver(source)

    def test_ollama_connection_error_includes_actual_cause(self) -> None:
        from unittest.mock import patch
        import requests

        with patch(
            "rag_orchestrator.requests.post",
            side_effect=requests.ConnectionError("connection refused"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "ConnectionError: connection refused"
            ):
                ollama_chat([{"role": "user", "content": "test"}])

    def test_ollama_http_error_includes_response_details(self) -> None:
        from unittest.mock import patch
        import requests

        response = requests.Response()
        response.status_code = 404
        response.url = "http://localhost:11434/api/chat"
        response._content = b'{"error":"model not found"}'
        with self.assertRaises(requests.HTTPError) as error:
            response.raise_for_status()

        with patch("rag_orchestrator.requests.post", side_effect=error.exception):
            with self.assertRaisesRegex(
                RuntimeError, "404 Client Error.*model not found"
            ):
                ollama_chat([{"role": "user", "content": "test"}])

    def test_gold_metrics_are_consumed_using_semantic_measure(self) -> None:
        silver = _build_silver(self.source)
        gold = _build_gold(silver)

        self.assertEqual(calculate_measure(gold, "total_spend", "CUST_99"), 400.0)
        self.assertEqual(calculate_measure(gold, "flagged_volume", "CUST_99"), 100.0)
        self.assertEqual(calculate_measure(gold, "transaction_count", None), 2)
        self.assertEqual(calculate_measure(gold, "average_amount", None), 200.0)

    def test_pending_amount_ranks_customers_for_requested_date(self) -> None:
        transactions = pd.DataFrame(
            [
                {
                    "transaction_id": 1,
                    "customer_id": "CUST_01",
                    "customer_name": "Northwind Traders",
                    "parent_company": "Northwind Group",
                    "amount": 150.0,
                    "status": "PENDING",
                    "transaction_date": "2026-08-05",
                    "region": "West",
                },
                {
                    "transaction_id": 2,
                    "customer_id": "CUST_01",
                    "customer_name": "Northwind Traders",
                    "parent_company": "Northwind Group",
                    "amount": 100.0,
                    "status": "PENDING",
                    "transaction_date": "2026-08-05",
                    "region": "West",
                },
                {
                    "transaction_id": 3,
                    "customer_id": "CUST_02",
                    "customer_name": "Contoso Retail",
                    "parent_company": "Contoso Group",
                    "amount": 400.0,
                    "status": "PENDING",
                    "transaction_date": "2026-08-05",
                    "region": "East",
                },
                {
                    "transaction_id": 4,
                    "customer_id": "CUST_03",
                    "customer_name": "Fabrikam Services",
                    "parent_company": "Fabrikam Group",
                    "amount": 1000.0,
                    "status": "FLAGGED",
                    "transaction_date": "2026-08-05",
                    "region": "Central",
                },
                {
                    "transaction_id": 5,
                    "customer_id": "CUST_04",
                    "customer_name": "Adventure Works",
                    "parent_company": "Adventure Group",
                    "amount": 900.0,
                    "status": "PENDING",
                    "transaction_date": "2026-08-06",
                    "region": "West",
                },
            ]
        )
        silver = _build_silver(transactions)

        result = calculate_top_customer(
            silver, "pending_amount", "2026-08-05", "PENDING"
        )

        self.assertEqual(
            result,
            {
                "customer_id": "CUST_02",
                "customer_name": "Contoso Retail",
                "amount": 400.0,
                "transaction_count": 1,
                "transaction_date": "2026-08-05",
                "status": "PENDING",
            },
        )

    def test_intent_extracts_date_status_and_top_customer_operation(self) -> None:
        from unittest.mock import patch

        question = "Which customer had higest pending amount on 2026-08-05"
        with patch("rag_orchestrator.ollama_chat") as ollama:
            intent = extract_query_intent(
                question,
                load_semantic_model(),
            )

        self.assertEqual(intent["measure"], "pending_amount")
        self.assertEqual(intent["transaction_date"], "2026-08-05")
        self.assertEqual(intent["status"], "PENDING")
        self.assertEqual(intent["operation"], "top_customer")
        ollama.assert_not_called()


if __name__ == "__main__":
    unittest.main()

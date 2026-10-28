# Iceberg Medallion + Graph RAG Demo

This local demo joins an Iceberg medallion pipeline, a semantic model, a Neo4j
knowledge graph, OPA authorization, and natural-language answers from Ollama.

## Architecture

- **Bronze** (`demo_bronze.transactions`): source transactions as ingested.
- **Silver** (`demo_silver.transactions`): normalized, deduplicated transactions
  with validation and a derived risk level.
- **Gold** (`demo_gold.customer_metrics`): customer-level finance metrics.
- **Semantic model** (`src/semantic_model.json`): names, definitions, and
  synonyms for metrics exposed to natural-language questions, including
  transaction-level pending amounts.
- **Knowledge graph**: customer-to-parent-company and customer-to-risk-profile
  links synchronized from Gold.
- **Logic and governance**: OPA allows `analytics_agent` to read the demo and
  allows `contractor_agent` to read only PUBLIC-classified data. A query that
  includes any HIGH-risk customer is classified as PII.
- **Consumption**: explicit ranked transaction questions are resolved against
  the semantic model and Silver transaction data, then enriched with graph
  context and checked by OPA. Other natural-language questions use Ollama to
  select a whitelisted metric and explain the computed result.

The demo uses deterministic synthetic finance transactions if
`data/source/transactions.csv` does not exist. To use your own CSV, provide the
columns `transaction_id,customer_id,customer_name,parent_company,amount,status,transaction_date,region`.
Valid statuses are `SETTLED`, `PENDING`, and `FLAGGED`. Existing source data is
not replaced by the generator.

## Run locally

From the repository root:

```powershell
docker compose -f infra\docker-compose.yml up -d
```

The Compose stack initializes the MinIO `warehouse` bucket and mounts the OPA
policy. Start your existing Ollama server separately and ensure it has the
configured model available. Defaults are `http://localhost:11434` and
`llama3.2:3b`; override them in PowerShell if needed:

```powershell
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:OLLAMA_MODEL = "llama3.2:3b"
```

Run the pipeline, synchronize Neo4j, and enter interactive query mode:

```powershell
win_env\Scripts\python.exe src\run_demo.py
```

Example questions:

- `What is Alpha Corp's flagged volume?`
- `How much total spend does CUST_99 have?`
- `What is the average transaction amount for CUST_01?`
- `Which customer had the highest pending amount on 2026-08-05?`

Date-filtered transaction metrics are calculated from Silver, where transaction
date and status are retained. Customer-level Gold metrics do not contain that
detail.

To demonstrate governance without natural-language metric extraction, an
explicit metric and customer can be supplied:

```powershell
win_env\Scripts\python.exe src\rag_orchestrator.py --question "Show flagged volume" --measure flagged_volume --customer-id CUST_99 --role contractor_agent
```

This request should be denied by OPA. Change the role to `analytics_agent` to
authorize it. Demo-owned Iceberg tables are refreshed on each pipeline run;
tables without the `lakehouse.demo=true` property are never overwritten.

Run the unit tests:

```powershell
win_env\Scripts\python.exe -m unittest discover -s tests
```

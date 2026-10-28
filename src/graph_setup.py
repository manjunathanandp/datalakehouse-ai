import os

from neo4j import GraphDatabase

from lakehouse_common import load_lakehouse_catalog, read_iceberg_table


def build_customer_graph(tx, customers: list[dict[str, object]]) -> None:
    tx.run(
        """
        UNWIND $customers AS row
        MERGE (c:Customer {id: row.customer_id})
        SET c.name = row.customer_name
        WITH c, row
        OPTIONAL MATCH (c)-[old]->()
        WHERE type(old) IN ["SUBSIDIARY_OF", "CATEGORIZED_AS"]
        WITH c, row, collect(old) AS stale_relationships
        FOREACH (relationship IN stale_relationships | DELETE relationship)
        MERGE (p:ParentCompany {name: row.parent_company})
        MERGE (r:RiskProfile {level: row.risk_level})
        MERGE (c)-[:SUBSIDIARY_OF]->(p)
        MERGE (c)-[:CATEGORIZED_AS]->(r)
        """,
        customers=customers,
    )


def sync_graph() -> None:
    catalog = load_lakehouse_catalog()
    gold = read_iceberg_table(catalog, ("demo_gold", "customer_metrics"))
    customers = gold[
        ["customer_id", "customer_name", "parent_company", "risk_level"]
    ].to_dict(orient="records")
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password123"),
        ),
    )
    try:
        with driver.session() as session:
            session.execute_write(build_customer_graph, customers)
    finally:
        driver.close()
    print(f"Knowledge graph synchronized for {len(customers)} customers.")


if __name__ == "__main__":
    sync_graph()

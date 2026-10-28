from graph_setup import sync_graph
from lakehouse_setup import run_pipeline
from rag_orchestrator import main as run_query_cli


def main() -> int:
    run_pipeline()
    sync_graph()
    return run_query_cli()


if __name__ == "__main__":
    raise SystemExit(main())

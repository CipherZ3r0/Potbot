"""
Prefect Flow — Wrapper around IngestionPipeline for workflow orchestration.
"""

import logging
import sys

from prefect import flow, task, get_run_logger

import config
from ingestion.pipeline import IngestionPipeline


@task(name="run-ingestion-pipeline", retries=2, retry_delay_seconds=10)
def task_run_pipeline(folder_path: str, recreate_index: bool = False) -> dict:
    logger = get_run_logger()
    logger.info(f"Running task: Ingesting '{folder_path}'")
    pipeline = IngestionPipeline()
    return pipeline.run(folder_path=folder_path, recreate_index=recreate_index)


@flow(name="securerag-ingest-flow", log_prints=True)
def ingest_flow(folder_path: str, recreate_index: bool = False) -> dict:
    """Prefect workflow wrapping IngestionPipeline."""
    return task_run_pipeline(folder_path=folder_path, recreate_index=recreate_index)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.prefect_flow <folder_path> [--recreate]")
        sys.exit(1)

    folder = sys.argv[1]
    recreate = "--recreate" in sys.argv
    result = ingest_flow(folder_path=folder, recreate_index=recreate)
    print(f"\nResult: {result}")

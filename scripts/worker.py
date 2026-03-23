"""Background worker for memharness slow path agents.

Runs entity extraction, summarization, and consolidation periodically.
Configure via environment variables:
  DATABASE_URL: Database connection string (default: sqlite:///memory.db)
  WORKER_INTERVAL: Seconds between runs (default: 300)

Usage:
  python scripts/worker.py

Docker:
  docker compose -f docker-compose.yml -f docker-compose.workers.yml up -d
"""

from __future__ import annotations

import asyncio
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("memharness.worker")


async def main() -> None:
    """Run slow path workers in a loop."""
    from memharness import MemoryHarness
    from memharness.core.slow_path import SlowPath

    db_url = os.environ.get("DATABASE_URL", "sqlite:///memory.db")
    interval = int(os.environ.get("WORKER_INTERVAL", "300"))

    harness = MemoryHarness(db_url)
    await harness.connect()
    slow = SlowPath(harness)

    logger.info("Worker started. DB=%s, interval=%ds", db_url, interval)

    while True:
        try:
            results = await slow.run_all()
            for r in results:
                logger.info(
                    "%s: processed=%d errors=%d duration=%.0fms",
                    r.worker,
                    r.processed,
                    r.errors,
                    r.duration_ms,
                )
        except Exception:
            logger.exception("Worker cycle failed")

        await asyncio.sleep(interval)


if __name__ == "__main__":
    asyncio.run(main())

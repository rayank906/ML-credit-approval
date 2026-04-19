from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import redis

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
for _path in (_BACKEND_DIR, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from database import Job, SessionLocal  # noqa: E402
from prediction import run_prediction  # noqa: E402

logger = logging.getLogger("backend.ml.redis_job_worker")


def _process_job(payload: dict[str, Any]) -> dict[str, Any]:
    r = run_prediction(payload)
    return {
        "decision": r["decision"],
        "credit_score": int(r["credit_score"]),
        "probability_risky": round(r["blended"] * 100, 1),
        "probability_safe": round((1 - r["blended"]) * 100, 1),
        "model_probability_risky": round(r["prob_unsafe"] * 100, 1),
        "confidence_score": r["confidence"],
        "strength": r["strength"],
        "top_factors": r["reasons"],
        "credit_breakdown": r["breakdown"],
        "risk_tips": r["tips"],
    }


def main() -> None:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    queue_name = os.environ.get("JOB_QUEUE_NAME", "ml_jobs")
    poll_timeout_s = int(os.environ.get("JOB_WORKER_POLL_TIMEOUT_S", "5"))

    r = redis.Redis.from_url(redis_url, decode_responses=True)
    logger.info("Worker started", extra={"queue": queue_name})

    while True:
        try:
            item = r.blpop(queue_name, timeout=poll_timeout_s)
            if not item:
                continue
            _queue, job_id = item

            with SessionLocal() as db:
                job = db.query(Job).filter(Job.id == job_id).first()
                if job is None:
                    logger.warning("Job not found", extra={"job_id": job_id})
                    continue

                job.status = "processing"
                job.updated_at = datetime.utcnow()
                db.commit()

                try:
                    result = _process_job(job.request or {})
                    job.result = result
                    job.error = None
                    job.status = "succeeded"
                except Exception as e:
                    job.result = None
                    job.error = str(e)
                    job.status = "failed"

                job.updated_at = datetime.utcnow()
                db.commit()
                logger.info("Job finished", extra={"job_id": job_id, "status": job.status})
        except KeyboardInterrupt:
            logger.info("Worker stopped")
            return
        except Exception as e:
            logger.error("Worker loop error", exc_info=e)
            time.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    main()

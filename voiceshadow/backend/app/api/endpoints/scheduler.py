"""
Scheduler management endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
import logging

from app.core.security import get_current_user

# Import scheduler service
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from services.scheduler_service import get_scheduler_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/jobs")
async def get_scheduled_jobs(
    current_user: dict = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get list of all scheduled jobs and their status"""
    try:
        scheduler = await get_scheduler_service()
        jobs = scheduler.get_jobs_info()
        return jobs
    except Exception as e:
        logger.error(f"Error getting scheduled jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get scheduled jobs")


@router.post("/jobs/{job_id}/trigger")
async def trigger_job(
    job_id: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, str]:
    """Manually trigger a specific scheduled job"""
    try:
        scheduler = await get_scheduler_service()
        await scheduler.trigger_job(job_id)
        return {"message": f"Job {job_id} triggered successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error triggering job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger job")


@router.post("/ingest-trends")
async def manual_trend_ingestion(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Manually trigger trend data ingestion"""
    try:
        scheduler = await get_scheduler_service()

        # Trigger the trend ingestion job
        await scheduler.trigger_job('trend_ingestion_2h')

        return {
            "message": "Trend ingestion started",
            "note": "Check logs for progress. Results will be available in a few minutes."
        }
    except Exception as e:
        logger.error(f"Error triggering trend ingestion: {e}")
        raise HTTPException(status_code=500, detail="Failed to start trend ingestion")
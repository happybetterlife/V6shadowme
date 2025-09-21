"""
Scheduler Service - Manages periodic tasks for data collection
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

import sys
import os
import importlib
import logging

logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def resolve_trend_ingestor():
    """Resolve trend ingestor with fallback chain."""
    candidates = [
        ("agents.enhanced_trend_ingestor", "EnhancedTrendIngestor"),
        ("agents.trend_ingestor", "TrendIngestor"),
    ]

    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            logger.info("Using %s.%s for trend ingestion", module_name, class_name)
            return cls
        except (ImportError, AttributeError) as exc:
            logger.debug("Cannot load %s.%s: %s", module_name, class_name, exc)

    raise ImportError("No trend ingestor implementation available")

TrendIngestor = resolve_trend_ingestor()


class SchedulerService:
    """Manages scheduled tasks for the application"""

    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.ingestor: Optional[TrendIngestor] = None
        self._running = False

    async def start(self):
        """Start the scheduler service"""
        if self._running:
            logger.warning("Scheduler already running")
            return

        try:
            # Initialize scheduler
            self.scheduler = AsyncIOScheduler()

            # Initialize trend ingestor
            self.ingestor = TrendIngestor()
            await self.ingestor.initialize()

            # Schedule jobs
            self._schedule_jobs()

            # Start scheduler
            self.scheduler.start()
            self._running = True

            logger.info("Scheduler service started successfully")

            # Run initial ingestion immediately
            asyncio.create_task(self._run_trend_ingestion())

        except Exception as e:
            logger.error(f"Failed to start scheduler service: {e}")
            raise

    def _schedule_jobs(self):
        """Configure all scheduled jobs"""

        # 1. Trend ingestion - every 2 hours
        self.scheduler.add_job(
            self._run_trend_ingestion,
            trigger=IntervalTrigger(hours=2),
            id='trend_ingestion_2h',
            name='Trend Data Ingestion (2 hours)',
            replace_existing=True,
            max_instances=1  # Prevent overlapping runs
        )

        # 2. Daily trend ingestion at specific time (e.g., 9 AM)
        self.scheduler.add_job(
            self._run_trend_ingestion,
            trigger=CronTrigger(hour=9, minute=0),
            id='trend_ingestion_daily',
            name='Daily Trend Data Ingestion',
            replace_existing=True,
            max_instances=1
        )

        # 3. Database cleanup - weekly (Sunday at 3 AM)
        self.scheduler.add_job(
            self._cleanup_old_data,
            trigger=CronTrigger(day_of_week='sun', hour=3, minute=0),
            id='db_cleanup_weekly',
            name='Weekly Database Cleanup',
            replace_existing=True
        )

        logger.info("Scheduled jobs configured:")
        for job in self.scheduler.get_jobs():
            logger.info(f"  - {job.name}: {job.trigger}")

    async def _run_trend_ingestion(self):
        """Execute trend data ingestion"""
        try:
            logger.info("Starting scheduled trend ingestion...")
            start_time = datetime.now()

            # Ensure ingestor is initialized
            if not self.ingestor:
                self.ingestor = TrendIngestor()
                await self.ingestor.initialize()

            # Run ingestion
            count = await self.ingestor.ingest_all()

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Trend ingestion completed: {count} new topics in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Error during trend ingestion: {e}")

    async def _cleanup_old_data(self):
        """Clean up old data from the database"""
        try:
            logger.info("Starting database cleanup...")

            if not self.ingestor:
                self.ingestor = TrendIngestor()
                await self.ingestor.initialize()

            # Delete topics older than 30 days
            # This would require adding a method to TrendsDatabase
            # For now, just log
            logger.info("Database cleanup completed (placeholder)")

        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")

    async def stop(self):
        """Stop the scheduler service"""
        if not self._running:
            return

        try:
            if self.scheduler:
                self.scheduler.shutdown(wait=True)

            if self.ingestor:
                await self.ingestor.close()

            self._running = False
            logger.info("Scheduler service stopped")

        except Exception as e:
            logger.error(f"Error stopping scheduler service: {e}")

    def get_jobs_info(self):
        """Get information about scheduled jobs"""
        if not self.scheduler:
            return []

        jobs_info = []
        for job in self.scheduler.get_jobs():
            jobs_info.append({
                'id': job.id,
                'name': job.name,
                'next_run': str(job.next_run_time) if job.next_run_time else None,
                'trigger': str(job.trigger)
            })

        return jobs_info

    async def trigger_job(self, job_id: str):
        """Manually trigger a specific job"""
        if not self.scheduler:
            raise RuntimeError("Scheduler not initialized")

        job = self.scheduler.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Run job immediately
        job.modify(next_run_time=datetime.now())
        logger.info(f"Manually triggered job: {job_id}")


# Global scheduler instance
scheduler_service = SchedulerService()


async def get_scheduler_service() -> SchedulerService:
    """Get the global scheduler service instance"""
    return scheduler_service
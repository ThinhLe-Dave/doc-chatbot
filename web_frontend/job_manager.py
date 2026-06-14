import threading
import uuid
from typing import Dict, Optional, Any


class Job:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status: str = "pending"
        self.progress: int = 0
        self.result: Optional[Dict] = None
        self.message: Optional[str] = None
        self.error: Optional[str] = None


class JobManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.jobs: Dict[str, Job] = {}
        return cls._instance

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self.jobs[job_id] = Job(job_id)
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self.jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        result: Optional[Dict] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
    ):
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if status is not None:
                    job.status = status
                if progress is not None:
                    job.progress = progress
                if result is not None:
                    job.result = result
                if message is not None:
                    job.message = message
                if error is not None:
                    job.error = error

    def to_dict(self, job: Job) -> Dict[str, Any]:
        return {
            "status": job.status,
            "progress": job.progress,
            "result": job.result,
            "message": job.message,
            "error": job.error,
        }

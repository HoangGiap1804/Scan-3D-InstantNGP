from pydantic import BaseModel
from typing import Optional

class TrainingRequest(BaseModel):
    job_id: str
    epochs: Optional[int] = 100

class ColmapRequest(BaseModel):
    job_id: str

class JobRequest(BaseModel):
    job_id: str

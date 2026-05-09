import os
from fastapi import APIRouter, BackgroundTasks, HTTPException
from api.db.repository import get_job_by_id, update_job_status
from api.models.requests import ColmapRequest
from api.tasks import run_colmap_task

router = APIRouter(prefix="/colmap")

@router.post("/start")
async def start_colmap(req: ColmapRequest, background_tasks: BackgroundTasks):
    job_id = req.job_id
    job = get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] == "processing":
        return {"message": "COLMAP is already running for this job"}

    target_dir = job["folder"]
    # Tìm file video trong thư mục target_dir
    video_files = [f for f in os.listdir(target_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        raise HTTPException(status_code=400, detail="No video file found in job folder")
    
    video_path = os.path.join(target_dir, video_files[0])
    
    background_tasks.add_task(run_colmap_task, video_path, target_dir, job_id)
    
    return {"status": "success", "message": "COLMAP processing started"}

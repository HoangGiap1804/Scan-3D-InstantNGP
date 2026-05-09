from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import os
import time
import shutil
from api.state import current_viewing_job
from api.db.repository import get_all_jobs, get_job_by_id, save_job
from api.ui import get_index_html

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def index():
    return get_index_html()

@router.get("/jobs")
async def get_all_jobs_route():
    all_jobs = get_all_jobs()
    # Chỉ trả về ID và ngày tạo để giảm dung lượng tải
    summary = [
        {"id": job["id"], "created_at": job["created_at"], "status": job["status"]} 
        for job in all_jobs.values()
    ]
    return {
        "jobs": summary,
        "current_viewing_job": current_viewing_job
    }


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    job = get_job_by_id(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return job

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    video_name = os.path.splitext(file.filename)[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job_id = f"{video_name}_{timestamp}"
    
    target_dir = os.path.join("videos", job_id)
    os.makedirs(target_dir, exist_ok=True)
    
    file_path = os.path.join(target_dir, file.filename)
    
    print(f"[UPLOAD] Saving {file.filename} to {file_path}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Chỉ lưu vào DB, không chạy colmap ngay
    save_job(job_id, file.filename, "uploaded", target_dir, "Video uploaded. Ready for COLMAP.")
    
    return {
        "status": "success",
        "message": "Upload complete. Ready for COLMAP processing.",
        "job_id": job_id,
        "endpoint": f"/status/{job_id}"
    }

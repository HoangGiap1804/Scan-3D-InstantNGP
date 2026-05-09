from fastapi import APIRouter, BackgroundTasks, HTTPException
from api.state import training_processes
from api.db.repository import get_job_by_id, update_job_status
from api.models.requests import TrainingRequest
from api.tasks import run_training_task
import os

router = APIRouter(prefix="/train")

@router.post("/start")
async def start_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    job_id = req.job_id
    job = get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_id in training_processes:
        return {"message": "Training is already running for this job"}

    if job["status"] == "uploaded":
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot start training. COLMAP status is '{job['status']}'. It must be 'completed' first."
        )


    workspace = job["folder"]
    data_path = workspace
    
    stop_flag = os.path.join(workspace, "stop.flag")
    if os.path.exists(stop_flag): os.remove(stop_flag)
    
    background_tasks.add_task(run_training_task, job_id, req.epochs, workspace, data_path)
    
    return {"status": "success", "message": f"Training started for {req.epochs} epochs"}

@router.post("/stop")
async def stop_training(req: TrainingRequest):
    import os
    job_id = req.job_id
    job = get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    workspace = job["folder"]
    stop_flag = os.path.join(workspace, "stop.flag")
    
    with open(stop_flag, "w") as f:
        f.write("stop")
        
    update_job_status(
        job_id, 
        training_status="stopped", 
        training_message="Stopping training (waiting for current epoch to finish)..."
    )
    
    return {"status": "success", "message": "Stop signal sent to training process"}

@router.post("/save")
async def save_model(req: TrainingRequest):
    import os
    job_id = req.job_id
    job = get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    workspace = job["folder"]
    save_flag = os.path.join(workspace, "save.flag")
    
    with open(save_flag, "w") as f:
        f.write("save")
        
    return {"status": "success", "message": "Save signal sent to training process"}

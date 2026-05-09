from fastapi import APIRouter, HTTPException
from api.state import viewer_process, current_viewing_job
from api.db.repository import get_job_by_id
from api.models.requests import TrainingRequest
from api.utils import get_local_ip

router = APIRouter(prefix="/view")

@router.post("/start")
async def start_viewer(req: TrainingRequest):
    import subprocess
    import os
    global viewer_process, current_viewing_job
    job_id = req.job_id
    job = get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    import api.state # Need to update the global in state.py
    
    if api.state.viewer_process and api.state.viewer_process.poll() is None:
        api.state.viewer_process.terminate()
        try:
            api.state.viewer_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api.state.viewer_process.kill()
            
    workspace = job["folder"]
    
    print(f"[VIEW] Khởi động viewer cho job {job_id} tại {workspace}")
    
    cmd = [
        "python3", "gui.py",
        "--workspace", workspace,
        "--res", "400",
        "--port_ws", "8000"
    ]
    
    try:
        api.state.viewer_process = subprocess.Popen(cmd)
        api.state.current_viewing_job = job_id
        return {"status": "success", "message": f"Viewer started for {job_id}", "ws_url": f"ws://{get_local_ip()}:8000"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start viewer: {str(e)}")

@router.post("/stop")
async def stop_viewer():
    import api.state
    if api.state.viewer_process and api.state.viewer_process.poll() is None:
        api.state.viewer_process.terminate()
        api.state.viewer_process = None
        api.state.current_viewing_job = None
        return {"status": "success", "message": "Viewer stopped"}
    return {"status": "success", "message": "No viewer was running"}

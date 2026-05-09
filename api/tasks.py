import subprocess
import re
from api.state import training_processes
from api.db.repository import update_job_status, add_metric

def run_colmap_task(video_path: str, target_dir: str, job_id: str):
    """
    Background task to process video:
    1. Extract frames using ffmpeg
    2. Run COLMAP to get camera poses
    3. Generate transforms.json
    """
    import os
    import time
    update_job_status(job_id, status="processing", message="Starting COLMAP processing...")
    print(f"[{job_id}] Processing started for {video_path}")
    
    start_time = time.time()
    
    try:
        # Construct command
        cmd = [
            "python3", "scripts/colmap2nerf.py",
            "--video", video_path,
            "--run_colmap",
            "--video_fps", "2",
            "--size", "800x800",
            "--yes"
        ]
        
        print(f"[{job_id}] Executing: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        for line in process.stdout:
            line = line.strip()
            if line:
                print(f"[{job_id}] {line}")
                if "running:" in line.lower():
                    update_job_status(job_id, message=line)
        
        process.wait()
        
        if process.returncode == 0:
            duration = time.time() - start_time
            result_msg = ""
            if os.path.exists(os.path.join(target_dir, "transforms.json")):
                result_msg = "transforms.json generated"
            elif os.path.exists(os.path.join(target_dir, "transforms_train.json")):
                 result_msg = "transforms_train.json generated"
            else:
                result_msg = "Processing finished but transforms file not found."
            
            update_job_status(
                job_id, 
                status="completed", 
                message=f"Finished successfully in {duration:.2f}s",
                completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                result=result_msg
            )
        else:
            update_job_status(job_id, status="failed", message=f"Process exited with code {process.returncode}")
            
    except Exception as e:
        update_job_status(job_id, status="error", message=str(e))

def run_training_task(job_id: str, epochs: int, workspace: str, data_path: str):
    """Background task to manage the training subprocess"""
    import time
    update_job_status(job_id, training_status="training", training_message=f"Starting training for {epochs} epochs...")
    
    cmd = [
        "python3", "train.py",
        "--path", data_path,
        "--workspace", workspace,
        "--epochs", str(epochs),
        "--fp16"
    ]
    
    metric_regex = re.compile(r"Epoch\s+(\d+)\s+complete\s+\|\s+loss:\s+([\d\.]+)\s+\|\s+PSNR\s+=\s+([\d\.]+)")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        training_processes[job_id] = process
        
        for line in process.stdout:
            line = line.strip()
            if line:
                if "Epoch" in line or "PSNR" in line:
                    update_job_status(job_id, training_message=line)
                    match = metric_regex.search(line)
                    if match:
                        add_metric(
                            job_id, 
                            epoch=int(match.group(1)), 
                            loss=float(match.group(2)), 
                            psnr=float(match.group(3))
                        )
        
        process.wait()
        
        if job_id in training_processes:
            del training_processes[job_id]
            
        if process.returncode == 0:
            update_job_status(
                job_id, 
                training_status="finished", 
                status="finished", 
                training_message="Training completed successfully."
            )
        else:
            # Note: We need a way to check current_training_status from DB if we want to be precise about "stopped"
            # For simplicity, we'll just check if it was marked as stopped in the DB recently
            # or just report failure.
            update_job_status(job_id, training_status="failed", training_message=f"Training failed with code {process.returncode}")
            
    except Exception as e:
        update_job_status(job_id, training_status="error", training_message=str(e))
        if job_id in training_processes:
            del training_processes[job_id]

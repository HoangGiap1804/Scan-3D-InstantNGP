import subprocess
from typing import Dict, Optional

# Track active training processes
# key: job_id, value: subprocess.Popen
training_processes: Dict[str, subprocess.Popen] = {}

# Track active viewer process (gui.py)
viewer_process: Optional[subprocess.Popen] = None
current_viewing_job: Optional[str] = None

from datetime import datetime
from api.db.session import get_db_connection

def save_job(job_id, filename, status, folder, message=""):
    conn = get_db_connection()
    cursor = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT OR REPLACE INTO jobs (id, filename, status, folder, message, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (job_id, filename, status, folder, message, created_at)
    )
    conn.commit()
    conn.close()

def update_job_status(job_id, status=None, message=None, training_status=None, training_message=None, completed_at=None, result=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    updates = []
    params = []
    
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if message is not None:
        updates.append("message = ?")
        params.append(message)
    if training_status is not None:
        updates.append("training_status = ?")
        params.append(training_status)
    if training_message is not None:
        updates.append("training_message = ?")
        params.append(training_message)
    if completed_at is not None:
        updates.append("completed_at = ?")
        params.append(completed_at)
    if result is not None:
        updates.append("result = ?")
        params.append(result)
        
    if updates:
        params.append(job_id)
        query = f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        
    conn.commit()
    conn.close()

def add_metric(job_id, epoch, loss, psnr):
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%H:%M:%S")
    cursor.execute(
        "INSERT INTO metrics (job_id, epoch, loss, psnr, timestamp) VALUES (?, ?, ?, ?, ?)",
        (job_id, epoch, loss, psnr, timestamp)
    )
    conn.commit()
    conn.close()

def get_all_jobs():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
    rows = cursor.fetchall()
    
    jobs_dict = {}
    for row in rows:
        job = dict(row)
        cursor.execute("SELECT epoch, loss, psnr, timestamp FROM metrics WHERE job_id = ? ORDER BY epoch ASC", (job['id'],))
        job['metrics'] = [dict(m) for m in cursor.fetchall()]
        jobs_dict[job['id']] = job
        
    conn.close()
    return jobs_dict

def get_job_by_id(job_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
        
    job = dict(row)
    cursor.execute("SELECT epoch, loss, psnr, timestamp FROM metrics WHERE job_id = ? ORDER BY epoch ASC", (job_id,))
    job['metrics'] = [dict(m) for m in cursor.fetchall()]
    conn.close()
    return job

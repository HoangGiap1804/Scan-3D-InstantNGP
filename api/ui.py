from fastapi.responses import HTMLResponse

def get_index_html():
    return """
    <html>
        <head>
            <title>Instant-NGP Server</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #e0e0e0; padding: 40px; display: flex; flex-direction: column; align-items: center; }
                .container { max-width: 900px; width: 100%; background: #1e1e1e; padding: 30px; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
                h1 { color: #4fc3f7; margin-top: 0; }
                .status-item { background: #2d2d2d; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #444; }
                .status-processing, .status-training { border-left-color: #ffb74d; }
                .status-completed, .status-finished { border-left-color: #81c784; }
                .status-failed, .status-error { border-left-color: #e57373; }
                .job-id { font-weight: bold; color: #4fc3f7; }
                .job-msg { font-size: 0.9em; color: #aaa; margin-top: 5px; }
                .controls { margin-top: 10px; display: flex; gap: 10px; }
                button { background: #4fc3f7; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; font-weight: bold; color: #000; }
                button.stop { background: #e57373; }
                button.save { background: #81c784; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Instant-NGP Processing Server</h1>
                <p>Status: <strong>Running</strong></p>
                <div id="jobs">
                    <h3>Recent Jobs</h3>
                    <div id="job-list">Loading jobs...</div>
                </div>
            </div>
            <script>
                async function control(action, jobId) {
                    const res = await fetch(`/train/${action}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ job_id: jobId })
                    });
                    const data = await res.json();
                    alert(data.message || data.error);
                }

                async function updateJobs() {
                    const res = await fetch('/jobs');
                    const data = await res.json();
                    const list = document.getElementById('job-list');
                    if (!data.jobs || Object.keys(data.jobs).length === 0) {
                        list.innerHTML = 'No jobs yet.';
                        return;
                    }
                    list.innerHTML = Object.entries(data.jobs).reverse().map(([id, job]) => `
                        <div class="status-item status-${job.status}">
                            <div class="job-id">${id}</div>
                            <div style="display: flex; justify-content: space-between;">
                                <div>
                                    <strong>COLMAP:</strong> ${job.status} <br>
                                    <span class="job-msg">${job.message}</span>
                                </div>
                                <div>
                                    <strong>Training:</strong> ${job.training_status || 'not started'} <br>
                                    <span class="job-msg">${job.training_message || '-'}</span>
                                </div>
                            </div>
                            <div class="controls">
                                <button onclick="control('start', '${id}')">Start Train</button>
                                <button class="stop" onclick="control('stop', '${id}')">Stop</button>
                                <button class="save" onclick="control('save', '${id}')">Save</button>
                            </div>
                        </div>
                    `).join('');
                }
                setInterval(updateJobs, 2000);
                updateJobs();
            </script>
        </body>
    </html>
    """

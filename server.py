import qrcode
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.utils import get_local_ip
from api.routes import jobs, colmap, train, view


app = FastAPI(title="Instant-NGP Processing Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(jobs.router)
app.include_router(colmap.router)
app.include_router(train.router)
app.include_router(view.router)


if __name__ == "__main__":
    local_ip = get_local_ip()
    port = 8080
    
    print("\n" + "="*50)
    print("INSTANT-NGP PROCESSING SERVER")
    print(f"Address: http://{local_ip}:{port}")
    print(f"Upload Endpoint: http://{local_ip}:{port}/upload")
    print("-" * 50)
    try:
        qr = qrcode.QRCode(version=1, box_size=1, border=1)
        qr.add_data(local_ip)
        qr.make(fit=True)
        print(f"Quét mã QR để lấy IP Server ({local_ip}):")
        qr.print_ascii(invert=True)
    except Exception as e:
        print(f"Could not generate QR code: {e}")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import subprocess
import os

app = FastAPI()

UPLOAD_DIR = "/app"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/status")
async def check_status():
    """Kiểm tra API có đang chạy không"""
    return {"status": "API is running"}

@app.head("/status")
async def head_status():
    """Trả về trạng thái API mà không có nội dung"""
    return {}

@app.post("/add_subtitles/")
async def add_subtitles(video: UploadFile = File(...), subtitle: UploadFile = File(...)):
    """Nhận video và subtitle, chèn sub bằng FFmpeg"""
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    subtitle_path = os.path.join(UPLOAD_DIR, subtitle.filename)
    output_path = os.path.join(UPLOAD_DIR, f"output_{video.filename}")

    with open(video_path, "wb") as f:
        f.write(video.file.read())

    with open(subtitle_path, "wb") as f:
        f.write(subtitle.file.read())

    # Chạy FFmpeg để gán sub
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"subtitles={subtitle_path}",
        "-c:a", "copy",
        output_path,
        "-y"  # Ghi đè nếu file đã tồn tại
    ]
    
    subprocess.run(command, check=True)

    return {"message": "Subtitles added!", "output_file": f"/download/output_{video.filename}"}

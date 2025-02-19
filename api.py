from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import subprocess
import os
import uvicorn
from fastapi.responses import FileResponse
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
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
async def attach_subtitles(
    video: UploadFile = File(...), 
    subtitle: UploadFile = File(...), 
    audio: UploadFile = File(...)
):
    """Nhận video, subtitle và audio, chèn sub vào video."""
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    subtitle_path = os.path.join(UPLOAD_DIR, subtitle.filename)
    audio_path = os.path.join(UPLOAD_DIR, audio.filename)
    output_path = os.path.join(UPLOAD_DIR, f"output_{video.filename}")

    # Lưu file vào thư mục
    with open(video_path, "wb") as f:
        f.write(video.file.read())
    with open(subtitle_path, "wb") as f:
        f.write(subtitle.file.read())
    with open(audio_path, "wb") as f:
        f.write(audio.file.read())

    # Load video
    video_clip = VideoFileClip(video_path)

    # Tạo subtitle clip
    generator = lambda txt: TextClip(txt, font='DejaVu-Sans', fontsize=40, color='white', stroke_color='black', stroke_width=1, method='caption', size=(video_clip.w * 0.9, None), align='center')
    subtitles = SubtitlesClip(subtitle_path, generator)

    # Ghép sub vào video
    video_with_subtitles = CompositeVideoClip([video_clip, subtitles.set_position(('center', 0.85), relative=True)])

    # Gán lại audio
    audio_clip = AudioFileClip(audio_path)
    video_with_subtitles = video_with_subtitles.set_audio(audio_clip)

    # Xuất video mới
    video_with_subtitles.write_videofile(output_path, codec="libx264", audio_codec="aac")

    # Trả về file video đã ghép sub
    return FileResponse(output_path, media_type="video/mp4", filename=f"output_{video.filename}")

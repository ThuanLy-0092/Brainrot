from fastapi import FastAPI, UploadFile, File
import yt_dlp
import random
import os
import whisper
from gtts import gTTS
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, AudioFileClip
from moviepy.video.tools.subtitles import SubtitlesClip

app = FastAPI()

def download_video_clip(url, output_path="video.mp4", duration=50):
    ydl_opts = {
        'format': 'bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]',
        'outtmpl': output_path,
        'download_sections': [f"*0-{duration}"],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def text_to_speech(text, output_file="output.mp3", playback_speed=1.5):
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(output_file)

    # Chỉnh tốc độ âm thanh
    audio = AudioSegment.from_file(output_file, format="mp3")
    modified_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * playback_speed)})
    modified_audio = modified_audio.set_frame_rate(audio.frame_rate)
    modified_audio.export(output_file, format="mp3")

    return output_file

def generate_subtitles(audio_path, output_srt="output.srt"):
    model = whisper.load_model("base")
    transcribe = model.transcribe(audio=audio_path, fp16=False)
    segments = transcribe["segments"]

    with open(output_srt, "w", encoding="utf-8") as f:
        for seg in segments:
            start = f"00:00:{int(seg['start']):02},000"
            end = f"00:00:{int(seg['end']):02},000"
            text = seg["text"].strip()
            segment_id = seg["id"] + 1
            f.write(f"{segment_id}\n{start} --> {end}\n{text}\n\n")
    
    return output_srt

def attach_subtitles(video_path, audio_path, output_video="final_output.mp4"):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    output_srt = generate_subtitles(audio_path)

    generator = lambda txt: TextClip(txt, font='DejaVu-Sans', fontsize=40, color='white', stroke_color='black',
                                     stroke_width=1, method='caption', size=(video.w * 0.9, None), align='center')
    subtitles = SubtitlesClip(output_srt, generator)
    video_with_subtitles = CompositeVideoClip([video, subtitles.set_position(('center', 0.85), relative=True)])
    video_with_subtitles = video_with_subtitles.set_audio(audio)
    video_with_subtitles.write_videofile(output_video, codec="libx264", audio_codec="aac")

    return output_video

@app.post("/process-video/")
async def process_video(url: str):
    video_path = "video.mp4"
    audio_path = "output.mp3"
    final_video = "final_output.mp4"

    # Download video
    download_video_clip(url, video_path, duration=50)

    # Chạy TTS
    text_to_speech("This is a sample text for testing", audio_path)

    # Thêm phụ đề vào video
    output_video = attach_subtitles(video_path, audio_path, final_video)

    return {"message": "Processing complete", "video_path": output_video}

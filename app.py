import streamlit as st
import yt_dlp
import random
import os
import whisper
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip
from moviepy.video.tools.subtitles import SubtitlesClip
from gtts import gTTS
from pydub import AudioSegment
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_random_video_from_playlist(playlist_url):
    ydl_opts = {'quiet': True, 'extract_flat': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    video_urls = [entry["url"] for entry in info["entries"] if "url" in entry]
    return random.choice(video_urls) if video_urls else None

def download_video_clip(url, output_path="video.mp4", duration=50):
    ydl_opts = {'format': 'best[ext=mp4]', 'outtmpl': output_path}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def generate_subtitles(audio_path, output_srt="output.srt"):
    model = whisper.load_model("base")
    transcribe = model.transcribe(audio=audio_path, fp16=False)
    segments = transcribe["segments"]
    with open(output_srt, "w", encoding="utf-8") as f:
        for seg in segments:
            start = f"0{int(seg['start'])},000"
            end = f"0{int(seg['end'])},000"
            text = seg["text"].strip()
            f.write(f"{seg['id'] + 1}\n{start} --> {end}\n{text}\n\n")
    return output_srt

def merge_subtitles(video_path, audio_path, subtitle_path, output_path="final_output.mp4"):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    generator = lambda txt: TextClip(txt, font='Arial', fontsize=40, color='white', stroke_color='black', stroke_width=1, method='caption', size=(video.w * 0.9, None), align='center')
    subtitles = SubtitlesClip(subtitle_path, generator)
    final_video = CompositeVideoClip([video, subtitles.set_position(('center', 0.85), relative=True)])
    final_video = final_video.set_audio(audio)
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path

st.title("📽️ Video Generator with Subtitles")
playlist_url = "https://www.youtube.com/playlist?list=PLJVvekmbcMxBCh1Cb997PA2hsrxmxdB6G"
pdf_file = st.file_uploader("Upload PDF File", type=["pdf"])

temp_pdf_path = "temp.pdf"
if pdf_file:
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_file.read())
    loader = PyPDFLoader(temp_pdf_path)
    combined_text = ' '.join([page.page_content for page in loader.load()])
    audio_path = "output.mp3"
    gTTS(text=combined_text, lang="en").save(audio_path)
    st.audio(audio_path)
    if st.button("Generate Video with Subtitles"):
        video_url = get_random_video_from_playlist(playlist_url)
        if video_url:
            video_path = download_video_clip(video_url, "video.mp4", duration=50)
            subtitle_path = generate_subtitles(audio_path, "output.srt")
            final_video = merge_subtitles(video_path, audio_path, subtitle_path, "final_output.mp4")
            st.video(final_video)
else:
    st.write("Upload a PDF to proceed!")

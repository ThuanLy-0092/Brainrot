import streamlit as st
import yt_dlp
import random
import re
import os
import whisper
import requests
from gtts import gTTS
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import timedelta
def get_random_video_from_playlist(playlist_url):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    video_urls = [entry["url"] for entry in info["entries"] if "url" in entry]
    return random.choice(video_urls) if video_urls else None

def download_video_clip(url, output_path="video.mp4", duration=50):
    ydl_opts = {
        'format': 'bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]',
        'outtmpl': output_path,
        'download_sections': [f"*0-{duration}"],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def preprocess_text(text):
    text = text.lower()
    text = ' '.join(text.split())
    text = re.sub(r'[^\w\s]', '', text)
    return text

def generate_brainrot_text(text):
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Transform the following text into a meme-filled brainrot version in **English only**. "
            "Strictly avoid Vietnamese words, names, or phrases. If any non-English words appear, replace them with relevant English equivalents. "
            "Keep it short, chaotic, but still understandable. "
            "Make it exaggerated and full of meme energy, using words like 'skibidi', 'mango', 'pizza', 'rizz', 'blox fruit', 'fortnite', 'sheeeesh', "
            "'oi oi oi', 'baka', 'gyat', 'what the hellll', and other fun slang. "
            "It **does not need to include all content**, but it must be **funny, short, and purely in English**. "
            "Do not use punctuation or special symbols. "
            "Do **not** say Here is the transformed text"
            "**Return only the transformed text, nothing else.** "
            "Here is the text to transform: {text}"
        ),
    )
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key="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",
        model_name="llama3-70b-8192"
    )
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())
    return chain.run({"text": text})

def text_to_speech(text, output_file="output.mp3", playback_speed=1.5):
    temp_mp3_path = "temp.mp3"

    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(temp_mp3_path)

    import time
    while not os.path.exists(temp_mp3_path):
        time.sleep(0.5)

    print("File exists:", os.path.exists(temp_mp3_path))

    audio = AudioSegment.from_file(temp_mp3_path, format="mp3")
    modified_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * playback_speed)})
    modified_audio = modified_audio.set_frame_rate(audio.frame_rate)

    modified_audio.export(output_file, format="mp3")
    return output_file

def process_video(video_path, audio_path, output_video="final_output.mp4"):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if os.path.getsize(video_path) == 0:
        raise ValueError("Downloaded video file is empty.")
    if os.path.getsize(audio_path) == 0:
        raise ValueError("Generated audio file is empty.")

    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    min_duration = min(video.duration, audio.duration)
    if min_duration <= 0:
        raise ValueError("Invalid video/audio duration.")

    video = video.subclip(0, min_duration)
    video = video.set_audio(audio)

    video.write_videofile(output_video, codec="libx264", audio_codec="aac")

    return output_video

def generate_srt_from_audio(audio_path, output_srt="output.srt"):
    if os.path.exists(output_srt):
        os.remove(output_srt)

    print("Generating subtitles using Whisper...")
    model = whisper.load_model("base")
    transcribe = model.transcribe(audio=audio_path, fp16=False)
    segments = transcribe["segments"]

    with open(output_srt, "a", encoding="utf-8") as f:
        for seg in segments:
            start = str(timedelta(seconds=int(seg["start"]))) + ",000"
            end = str(timedelta(seconds=int(seg["end"]))) + ",000"
            text = seg["text"].strip()
            segment_id = seg["id"] + 1
            f.write(f"{segment_id}\n{start} --> {end}\n{text}\n\n")

    print("Subtitles generated successfully:", output_srt)
    return output_srt


def add_subtitles_to_video(video_path, subtitle_path, audio_path, output_video="final_video_with_subs.mp4"):
    api_url = "https://brainrot-fkem.onrender.com/add_subtitles/"

    with open(video_path, "rb") as vid, open(subtitle_path, "rb") as sub, open(audio_path, "rb") as aud:
        files = {
            "video": vid,
            "subtitle": sub,
            "audio": aud
        }
        response = requests.post(api_url, files=files)

    if response.status_code == 200:
        with open(output_video, "wb") as f:
            f.write(response.content)  # Lưu video nhận được từ API
        
        return output_video
    else:
        raise Exception(f"Failed to attach subtitles: {response.text}")



st.title("📽️ Video Generator from YouTube & PDF")
playlist_url = "https://www.youtube.com/playlist?list=PLJVvekmbcMxBCh1Cb997PA2hsrxmxdB6G"
pdf_file = st.file_uploader("Upload PDF File", type=["pdf"])

temp_pdf_path = "temp.pdf"

if pdf_file:
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_file.read())
    
    loader = PyPDFLoader(temp_pdf_path)
    pages = loader.load()[:10]  # Chỉ lấy 10 trang đầu tiên
    combined_text = ' '.join(page.page_content for page in pages)
    
    cleaned_text = preprocess_text(combined_text)
    brainrot_text = generate_brainrot_text(cleaned_text)
    
    if st.button("Generate Video"):
        random_video_url = get_random_video_from_playlist(playlist_url)
        if random_video_url:
            audio_path = text_to_speech(brainrot_text, "output.mp3", playback_speed=1.5)
            video_path = download_video_clip(random_video_url, "video.mp4", duration=50)
            final_video = process_video(video_path, audio_path, "final_output.mp4")
    
            # 🎯 Tạo file SRT từ audio bằng Whisper
            subtitle_path = generate_srt_from_audio(audio_path, "output.srt")
    
            # 🚀 Gửi video + sub lên API
            final_video_with_subs = add_subtitles_to_video(final_video, subtitle_path, audio_path)
    
            st.video(final_video_with_subs)

else:
    st.write("Upload a PDF to proceed!")

import streamlit as st
import yt_dlp
import random
import re
import os
import whisper
from gtts import gTTS
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, AudioFileClip
from moviepy.video.tools.subtitles import SubtitlesClip
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
        'format': 'best[ext=mp4]',
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
    
    audio = AudioSegment.from_file(temp_mp3_path, format="mp3")
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
    
    generator = lambda txt: TextClip(
        txt, font='Arial', fontsize=40, color='white', stroke_color='black',
        stroke_width=1, method='caption', size=(video.w * 0.9, None), align='center'
    )
    subtitles = SubtitlesClip(output_srt, generator)
    video_with_subtitles = CompositeVideoClip([video, subtitles.set_position(('center', 0.85), relative=True)])
    video_with_subtitles = video_with_subtitles.set_audio(audio)
    video_with_subtitles.write_videofile(output_video, codec="libx264", audio_codec="aac")
    
    return output_video

st.title("📽️ Video Generator from YouTube & PDF")
playlist_url = "https://www.youtube.com/playlist?list=PLJVvekmbcMxBCh1Cb997PA2hsrxmxdB6G"
pdf_file = st.file_uploader("Upload PDF File", type=["pdf"])

temp_pdf_path = "temp.pdf"
if pdf_file:
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_file.read())
    loader = PyPDFLoader(temp_pdf_path)
    pages = [page.page_content for page in loader.load()]
    combined_text = ' '.join(pages)
    cleaned_text = preprocess_text(combined_text)
    brainrot_text = generate_brainrot_text(cleaned_text)
    audio_path = text_to_speech(brainrot_text, "output.mp3", playback_speed=1.5)
    st.audio(audio_path)
    if st.button("Generate Video"):
        random_video_url = get_random_video_from_playlist(playlist_url)
        if random_video_url:
            video_path = download_video_clip(random_video_url, "video.mp4", duration=50)
            final_video = attach_subtitles(video_path, audio_path, "final_output.mp4")
            st.video(final_video)
else:
    st.write("Upload a PDF to proceed!")

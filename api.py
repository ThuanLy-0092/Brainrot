from fastapi import FastAPI, UploadFile, File
import yt_dlp
import os
import random
import whisper
import re
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

app = FastAPI()

### ======================= 1. Tải video từ YouTube ======================= ###
def download_video_clip(url, output_path="video.mp4", duration=50):
    ydl_opts = {
        'format': 'bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]',
        'outtmpl': output_path,
        'download_sections': [f"*0-{duration}"],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

### ======================= 2. Đọc PDF và xử lý văn bản ======================= ###
def preprocess_text(text):
    text = text.lower()
    text = ' '.join(text.split())
    text = re.sub(r'[^\w\s]', '', text)
    return text

def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = [page.page_content for page in loader.load()]
    combined_text = ' '.join(pages)
    return preprocess_text(combined_text)

### ======================= 3. Tạo "Brainrot Text" bằng LLM ======================= ###
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

### ======================= 4. Chuyển thành giọng nói (TTS) ======================= ###
def text_to_speech(text, output_file="output.mp3", playback_speed=1.5):
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(output_file)

    # Chỉnh tốc độ âm thanh
    audio = AudioSegment.from_file(output_file, format="mp3")
    modified_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * playback_speed)})
    modified_audio = modified_audio.set_frame_rate(audio.frame_rate)
    modified_audio.export(output_file, format="mp3")

    return output_file

### ======================= 5. Tạo phụ đề từ giọng nói ======================= ###
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

### ======================= 6. Ghép video, âm thanh và phụ đề ======================= ###
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

### ======================= 7. API Process ======================= ###
@app.post("/process-video/")
async def process_video(url: str, pdf: UploadFile = File(...)):
    video_path = "video.mp4"
    audio_path = "output.mp3"
    final_video = "final_output.mp4"
    temp_pdf_path = "temp.pdf"

    # Lưu PDF tạm thời
    with open(temp_pdf_path, "wb") as f:
        f.write(await pdf.read())

    # Trích xuất nội dung từ PDF
    extracted_text = extract_text_from_pdf(temp_pdf_path)

    # Tạo brainrot text
    brainrot_text = generate_brainrot_text(extracted_text)

    # Chuyển văn bản thành giọng nói
    text_to_speech(brainrot_text, audio_path)

    # Tải video từ YouTube
    download_video_clip(url, video_path, duration=50)

    # Thêm phụ đề vào video
    output_video = attach_subtitles(video_path, audio_path, final_video)

    return {"message": "Processing complete", "video_path": output_video}


import os, uuid, pathlib, tempfile, subprocess, datetime, requests, srt, streamlit as st
import whisper
from TTS.api import TTS
from moviepy.editor import (VideoFileClip, TextClip, CompositeVideoClip,
                            concatenate_videoclips, AudioFileClip)

st.set_page_config(page_title="Free Faceless Reels", layout="centered")
OUTPUT_DIR = pathlib.Path("renders")
OUTPUT_DIR.mkdir(exist_ok=True)

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource
def load_tts():
    return TTS("tts_models/en/ljspeech/tacotron2-DDC")

def generate_script(prompt):
    API_URL = "https://api-inference.huggingface.co/models/facebook/opt-1.3b"
    payload = {"inputs": f"Write a 70-word Instagram Reel script about: {prompt}. Hook first 8 words."}
    txt = requests.post(API_URL, json=payload).json()[0]["generated_text"]
    return " ".join(txt.split()[:70]) + " Follow for more!"

def fetch_broll():
    headers = {"Authorization": "563492ad6f9170000100000153c40a3a7f5c4c06abcb8b8b6cf1b9c5"}
    r = requests.get("https://api.pexels.com/videos/search",
                     params={"query": "motivation,business", "per_page": 6}, headers=headers).json()
    local = []
    for i, v in enumerate(r["videos"]):
        url = v["video_files"][0]["link"]
        tmp = pathlib.Path(tempfile.gettempdir()) / f"clip{i}.mp4"
        if not tmp.exists():
            tmp.write_bytes(requests.get(url).content)
        local.append(str(tmp))
    return local

def create_reel(prompt):
    script = generate_script(prompt)
    tts = load_tts()
    voice = pathlib.Path(tempfile.gettempdir()) / "voice.wav"
    tts.tts_to_file(text=script, file_path=str(voice))

    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(voice)]
    ).decode().split('"duration":')[1].split("}")[0])

    clips = fetch_broll()
    v_clips = [VideoFileClip(c).resize(height=1920).crop(x_center=540, width=1080, height=1920).subclip(0, 5)
               for c in clips]
    composed = concatenate_videoclips(v_clips, method="compose").set_audio(AudioFileClip(str(voice)))

    model = load_whisper()
    segments = model.transcribe(str(voice))["segments"]
    subs = [srt.Subtitle(index=i,
                         start=datetime.timedelta(seconds=s["start"]),
                         end=datetime.timedelta(seconds=s["end"]),
                         content=s["text"].upper()) for i, s in enumerate(segments)]
    subs_clips = [TextClip(txt=sub.content, fontsize=70, color="white", font="DejaVu-Sans-Bold",
                           size=(1080, 200), bg_color="transparent")
                  .set_position(("center", "center"))
                  .set_start(sub.start.total_seconds())
                  .set_duration((sub.end - sub.start).total_seconds())
                  for sub in subs]
    final = CompositeVideoClip([composed] + subs_clips)
    out = OUTPUT_DIR / f"reel_{uuid.uuid4().hex[:8]}.mp4"
    final.write_videofile(str(out), fps=24, codec="libx264", audio_codec="aac", logger=None)
    return out, script

st.title("🎞️ Free Faceless Reels Generator")
prompt = st.text_area("What’s your reel about?", "Productivity tips for entrepreneurs")
if st.button("Generate Reel"):
    with st.spinner("Rendering on CPU … ~3-5 min"):
        path, scr = create_reel(prompt)
    st.success("Done!")
    st.video(str(path))
    with open(path, "rb") as f:
        st.download_button("Download MP4", f, file_name=path.name)
    with st.expander("Script"):
        st.text(scr)

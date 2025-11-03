# streamlit_app.py  –  free faceless-reel generator
import os, uuid, pathlib, tempfile, subprocess, datetime, requests, srt
import streamlit as st, whisper
from TTS.api import TTS
from moviepy.editor import *

st.set_page_config("Free Reels", "🎞️", "centered")
REND = pathlib.Path("renders"); REND.mkdir(exist_ok=True)

@st.cache_resource
def load_whisper(): return whisper.load_model("base")
@st.cache_resource
def load_tts(): return TTS("tts_models/en/ljspeech/tacotron2-DDC")

def script(prompt):
    r = requests.post("https://api-inference.huggingface.co/models/facebook/opt-1.3b",
                      json={"inputs": f"70-word IG reel about: {prompt}. Hook first 8 words."}).json()
    txt = r[0]["generated_text"]
    return " ".join(txt.split()[:70]) + " Follow for more!"

def broll():
    hdr = {"Authorization": "563492ad6f9170000100000153c40a3a7f5c4c06abcb8b8b6cf1b9c5"}
    r = requests.get("https://api.pexels.com/videos/search",
                     params={"query": "motivation,business", "per_page": 6}, headers=hdr).json()
    loc = []
    for i, v in enumerate(r["videos"]):
        url = v["video_files"][0]["link"]
        tmp = pathlib.Path(tempfile.gettempdir()) / f"clip{i}.mp4"
        if not tmp.exists(): tmp.write_bytes(requests.get(url).content)
        loc.append(str(tmp))
    return loc

def reel(prompt):
    scr = script(prompt)
    tts = load_tts()
    voc = pathlib.Path(tempfile.gettempdir()) / "voice.wav"
    tts.tts_to_file(text=scr, file_path=str(voc))
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(voc)]
    ).decode().split('"duration":')[1].split("}")[0])
    clips = broll()
    vids = [VideoFileClip(c).resize(height=1920).crop(x_center=540, width=1080, height=1920).subclip(0, 5)
            for c in clips]
    comp = concatenate_videoclips(vids, method="compose").set_audio(AudioFileClip(str(voc)))
    segs = load_whisper().transcribe(str(voc))["segments"]
    subs = [srt.Subtitle(index=i,
                         start=datetime.timedelta(seconds=s["start"]),
                         end=datetime.timedelta(seconds=s["end"]),
                         content=s["text"].upper()) for i, s in enumerate(segs)]
    subs_clip = [TextClip(txt=sub.content, fontsize=70, color="white", font="DejaVu-Sans-Bold",
                          size=(1080, 200), bg_color="transparent")
                 .set_position(("center", "center"))
                 .set_start(sub.start.total_seconds())
                 .set_duration((sub.end - sub.start).total_seconds())
                 for sub in subs]
    final = CompositeVideoClip([comp] + subs_clip)
    out = REND / f"reel_{uuid.uuid4().hex[:8]}.mp4"
    final.write_videofile(str(out), fps=24, codec="libx264", audio_codec="aac", logger=None)
    return out, scr

st.title("🎞️ Free Faceless Reels Generator")
p = st.text_area("What’s your reel about?", "Productivity tips for entrepreneurs")
if st.button("Generate Reel"):
    with st.spinner("Rendering on CPU … ~3-5 min"):
        path, script = reel(p)
    st.success("Done!"); st.video(str(path))
    with open(path, "rb") as f: st.download_button("Download MP4", f, file_name=path.name)
    with st.expander("Script"): st.text(script)

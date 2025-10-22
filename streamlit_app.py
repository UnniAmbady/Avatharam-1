# streamlit_app.py
# Single-screen mobile UI. Startup sequence matches Stage-1 Ver.7:
# 1) POST /v1/streaming.new  -> session_id, offer.sdp, ice servers
# 2) POST /v1/streaming.create_token  (with session_id) -> token
# 3) time.sleep(1.0)
# 4) Render viewer.html via components.html(...) with placeholders replaced
#
# Mic Echo (optional): Start/Stop uses streamlit-webrtc. On Start, mic audio is captured
# and transcribed (OpenAI Whisper if key present) then sent back to the SAME session to speak.
# If you only need avatar startup (steps 1-4), you can comment out the webrtc / echo parts.

import os
import json
import time
from pathlib import Path
from typing import Optional

import requests
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Optional STT libs only used if OPENAI_API_KEY is present
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
    _HAS_WEBRTC = True
except Exception:
    _HAS_WEBRTC = False

# -------------------------------
# Page & Mobile CSS
# -------------------------------
st.set_page_config(page_title="Alessandra • Echo", page_icon="🗣️", layout="centered")
st.markdown(
    """
    <style>
      .block-container { padding-top: 0.6rem; padding-bottom: 1.6rem; }
      .stButton > button { width: 100%; height: 56px; font-size: 1.05rem; border-radius: 12px; }
      .caption { color:#64748b; font-size:0.8rem; }
      iframe { border:none; border-radius:14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Secrets helpers (supports [HeyGen] or [heygen], env fallback)
# -------------------------------
def get_heygen_key() -> Optional[str]:
    s = st.secrets
    for sec, k in [("HeyGen", "heygen_api_key"), ("heygen", "heygen_api_key")]:
        try:
            return s[sec][k]
        except Exception:
            pass
    return os.getenv("HEYGEN_API_KEY")

HEYGEN_API_KEY = get_heygen_key()
if not HEYGEN_API_KEY:
    st.error(
        "Missing HeyGen API key. Add to `.streamlit/secrets.toml`:\n\n"
        "[HeyGen]\nheygen_api_key = \"NzBiOTA…==\""
    )
    st.stop()

OPENAI_API_KEY = (st.secrets.get("openai", {}) or {}).get("api_key") or os.getenv("OPENAI_API_KEY")

AVATAR_ID = "Alessandra_CasualLook_public"
VOICE_ID  = "0d3f35185d7c4360b9f03312e0264d59"

BASE = "https://api.heygen.com/v1"
API_STREAM_NEW   = f"{BASE}/streaming.new"
API_CREATE_TOKEN = f"{BASE}/streaming.create_token"

HEADERS = {
    "x-api-key": HEYGEN_API_KEY,
    "accept": "application/json",
    "Content-Type": "application/json",
}

# -------------------------------
# Step 1: create session (new)
# -------------------------------
def new_session(avatar_id: str, voice_id: Optional[str] = None):
    payload = {"avatar_id": avatar_id}
    if voice_id:
        payload["voice_id"] = voice_id

    r = requests.post(API_STREAM_NEW, headers=HEADERS, json=payload, timeout=45)
    r.raise_for_status()
    d = r.json()["data"]

    session_id = d["session_id"]
    # some tenants return {"offer":{"sdp":...}}, others {"sdp":{"sdp":...}}
    offer_sdp = (d.get("offer") or d.get("sdp") or {}).get("sdp")
    if not offer_sdp:
        raise RuntimeError("Missing offer.sdp in streaming.new response")

    ice2 = d.get("ice_servers2") or []
    ice1 = d.get("ice_servers") or []
    rtc_config = {"iceServers": (ice2 or ice1 or [{"urls": ["stun:stun.l.google.com:19302"]}])}
    return session_id, offer_sdp, rtc_config

# -------------------------------
# Step 2: create token for that session
# -------------------------------
def create_session_token(session_id: str) -> str:
    r = requests.post(API_CREATE_TOKEN, headers=HEADERS, json={"session_id": session_id}, timeout=45)
    r.raise_for_status()
    dd = r.json()["data"]
    return dd.get("token") or dd.get("access_token")

# -------------------------------
# Speak via REST (same session). Endpoint names can vary by tenant;
# we try two common paths; no hard fail if both 404.
# -------------------------------
def speak_text(session_id: str, text: str):
    for path in (
        f"{BASE}/streaming/session/{session_id}/speak",
        f"{BASE}/streaming.input",  # legacy multi-session input
    ):
        try:
            payload = {"text": text, "session_id": session_id}  # harmless extra field for first path
            r = requests.post(path, headers=HEADERS, json=payload, timeout=20)
            if 200 <= r.status_code < 300:
                return True
        except Exception:
            pass
    st.toast("Speak endpoint not accepted; check your org path.")
    return False

# -------------------------------
# Render viewer.html (Step 4)
# -------------------------------
def render_viewer_html(token: str, session_id: str, offer_sdp: str, rtc_config: dict, height: int = 600):
    viewer_path = Path(__file__).parent / "viewer.html"
    if not viewer_path.exists():
        st.error("viewer.html not found next to streamlit_app.py. Please add it from Stage-1.")
        st.stop()

    raw = viewer_path.read_text(encoding="utf-8")
    # Keep SDP newlines unescaped; we inject without quotes by trimming json dumps quoting.
    sdp_raw = json.dumps(offer_sdp)[1:-1]
    html = (
        raw.replace("__SESSION_TOKEN__", token)
           .replace("__SESSION_ID__", session_id)
           .replace("__AVATAR_NAME__", "Alessandra Casual Look")
           .replace("__OFFER_SDP__", sdp_raw)
           .replace("__RTC_CONFIG__", json.dumps(rtc_config))
    )
    components.html(html, height=height, scrolling=False)

# -------------------------------
# Optional: Mic → STT → Speak back (Echo)
# -------------------------------
class _MicProc(AudioProcessorBase):
    def __init__(self):
        self.buf = b""
        self.sample_rate = 16000

    def recv_audio(self, frame):
        pcm16 = frame.to_ndarray(format="s16")
        if pcm16.ndim == 2 and pcm16.shape[0] > 1:
            pcm16 = pcm16[0:1, :]
        pcm16 = np.squeeze(pcm16)
        in_rate = frame.sample_rate
        if in_rate != self.sample_rate:
            # quick linear resample
            dur = pcm16.shape[0] / in_rate
            new_len = int(dur * self.sample_rate)
            pcm16 = np.interp(
                np.linspace(0, pcm16.shape[0], new_len, endpoint=False),
                np.arange(pcm16.shape[0]),
                pcm16.astype(np.float32),
            ).astype(np.int16)
        self.buf += pcm16.tobytes()
        return frame

def _transcribe(pcm: bytes, rate: int) -> Optional[str]:
    if not OPENAI_API_KEY or not pcm:
        return None
    try:
        from openai import OpenAI
        import tempfile, wave
        client = OpenAI(api_key=OPENAI_API_KEY)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(model="whisper-1", file=f)
        text = getattr(resp, "text", "") or ""
        return text.strip() or None
    except Exception as e:
        st.toast(f"STT error: {e}")
        return None

# -------------------------------
# UI
# -------------------------------
st.markdown("### Alessandra • Echo mode")
with st.spinner("Starting avatar…"):
    # Step 1
    session_id, offer_sdp, rtc_config = new_session(AVATAR_ID, VOICE_ID)
    # Step 2
    token = create_session_token(session_id)
    # Step 3 (tiny but important)
    time.sleep(1.0)

# Step 4
render_viewer_html(token, session_id, offer_sdp, rtc_config, height=600)

# Controls (optional Echo)
if "webrtc_ctx" not in st.session_state:
    st.session_state.webrtc_ctx = None
if "mic_proc" not in st.session_state:
    st.session_state.mic_proc = _MicProc()

col1, col2 = st.columns(2, gap="small")
with col1:
    start_clicked = st.button("▶️  Start", type="primary")
with col2:
    stop_clicked = st.button("⏹️  Stop")

if start_clicked and _HAS_WEBRTC:
    st.session_state.webrtc_ctx = webrtc_streamer(
        key="mic-only",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=lambda: st.session_state.mic_proc,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=False,
    )
    st.toast("Mic started. Speak to echo through the avatar.")

if stop_clicked:
    # Proper shutdown (fixes AttributeError: no .destroy())
    ctx = st.session_state.get("webrtc_ctx")
    if ctx and getattr(ctx, "state", None) and ctx.state.playing:
        ctx.stop()
    st.session_state.webrtc_ctx = None

# Echo loop trigger (simple: every click of 'Start' we try transcribing what has accumulated so far)
# For continuous echo you may set a timer/interval in JS or run a small loop with st.autorefresh.
if start_clicked and OPENAI_API_KEY:
    text = _transcribe(st.session_state.mic_proc.buf, st.session_state.mic_proc.sample_rate)
    st.session_state.mic_proc.buf = b""
    if text:
        speak_text(session_id, text)
        st.caption(f"You said: “{text}” → echoed.")
elif start_clicked and not OPENAI_API_KEY:
    st.caption("Tip: add [openai].api_key in secrets to enable voice echo.")
    
st.markdown("<div class='caption'>Open in Safari/Chrome on iPhone. Allow microphone access.</div>", unsafe_allow_html=True)

# streamlit_app.py — Diagnostic build
# Sequence:
# 1) POST /v1/streaming.new  -> session_id, offer.sdp, ice servers
# 2) POST /v1/streaming.create_token (with session_id) -> token
# 3) sleep(1.0)
# 4) Render local viewer.html (now base64-SDP + verbose diagnostics)

import os
import json
import time
import base64
from pathlib import Path
from typing import Optional

import requests
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Optional STT for Echo (only if you add OPENAI_API_KEY)
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
    _HAS_WEBRTC = True
except Exception:
    _HAS_WEBRTC = False

# -------------------------------
# Page & Mobile CSS
# -------------------------------
st.set_page_config(page_title="Alessandra • Echo (diag)", page_icon="🗣️", layout="centered")
st.markdown(
    """
    <style>
      .block-container { padding-top: 0.4rem; padding-bottom: 1.2rem; }
      .stButton > button { width: 100%; height: 56px; font-size: 1.05rem; border-radius: 12px; }
      .caption { color:#64748b; font-size:0.8rem; }
      iframe { border:none; border-radius:14px; }
      /* Hide the internal Start button of streamlit-webrtc */
      div.st-webrtc > div:has(button) { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Secrets helpers
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
    st.error("Missing HeyGen API key. Add to `.streamlit/secrets.toml`:\n\n[HeyGen]\nheygen_api_key = \"…\"")
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

# UI toggles
with st.expander("Diagnostics"):
    VERBOSE = st.checkbox("Verbose logs (print to server logs & UI)", value=True)
    st.caption("If something stalls, copy the UI log and server log for support.")

def log(msg: str):
    if VERBOSE:
        st.write("▪️", msg)
    print(msg, flush=True)

# -------------------------------
# Step 1: new session
# -------------------------------
def new_session(avatar_id: str, voice_id: Optional[str] = None):
    payload = {"avatar_id": avatar_id}
    if voice_id:
        payload["voice_id"] = voice_id

    log(f"[1] POST {API_STREAM_NEW}")
    r = requests.post(API_STREAM_NEW, headers=HEADERS, json=payload, timeout=45)
    log(f"[1] status={r.status_code}")
    if not r.ok:
        log(f"[1] body={r.text}")
        r.raise_for_status()
    d = r.json().get("data", {})
    session_id = d.get("session_id")
    offer_obj = d.get("offer") or d.get("sdp") or {}
    offer_sdp = offer_obj.get("sdp")
    ice2 = d.get("ice_servers2") or []
    ice1 = d.get("ice_servers") or []
    rtc_config = {"iceServers": (ice2 or ice1 or [{"urls": ["stun:stun.l.google.com:19302"]}])}
    if not session_id or not offer_sdp:
        raise RuntimeError("Missing session_id or offer.sdp in streaming.new response")
    return session_id, offer_sdp, rtc_config

# -------------------------------
# Step 2: session token
# -------------------------------
def create_session_token(session_id: str) -> str:
    body = {"session_id": session_id}
    log(f"[2] POST {API_CREATE_TOKEN}")
    r = requests.post(API_CREATE_TOKEN, headers=HEADERS, json=body, timeout=45)
    log(f"[2] status={r.status_code}")
    if not r.ok:
        log(f"[2] body={r.text}")
        r.raise_for_status()
    dd = r.json().get("data", {})
    token = dd.get("token") or dd.get("access_token")
    if not token:
        raise RuntimeError("No token/access_token returned by streaming.create_token")
    return token

# -------------------------------
# Render viewer.html
# -------------------------------
def render_viewer_html(token: str, session_id: str, offer_sdp: str, rtc_config: dict, height: int = 600):
    viewer_path = Path(__file__).parent / "viewer.html"
    if not viewer_path.exists():
        st.error("viewer.html not found next to streamlit_app.py.")
        st.stop()

    # Pass SDP as base64 to avoid any quoting/newline issues
    sdp_b64 = base64.b64encode(offer_sdp.encode("utf-8")).decode("ascii")

    raw = viewer_path.read_text(encoding="utf-8")
    html = (
        raw.replace("__SESSION_TOKEN__", token)
           .replace("__SESSION_ID__", session_id)
           .replace("__AVATAR_NAME__", "Alessandra Casual Look")
           .replace("__OFFER_SDP_B64__", sdp_b64)
           .replace("__RTC_CONFIG__", json.dumps(rtc_config))
           .replace("__API_BASE__", BASE)
    )
    components.html(html, height=height, scrolling=False)

# -------------------------------
# Optional Echo (mic → STT → speak)
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
            import numpy as _np
            dur = pcm16.shape[0] / in_rate
            new_len = int(dur * self.sample_rate)
            pcm16 = _np.interp(
                _np.linspace(0, pcm16.shape[0], new_len, endpoint=False),
                _np.arange(pcm16.shape[0]),
                pcm16.astype(_np.float32),
            ).astype(np.int16)
        self.buf += pcm16.tobytes()
        return frame

def speak_text(session_id: str, text: str):
    # Try both common endpoints/headers and report exactly what worked
    tried = []

    # 1) Bearer token endpoint path
    url1 = f"{BASE}/streaming/session/{session_id}/speak"
    hdr1 = {"Authorization": f"Bearer {st.session_state.get('HG_TOKEN','')}",
            "accept": "application/json", "Content-Type": "application/json"}
    tried.append(("Bearer", url1))
    r1 = requests.post(url1, headers=hdr1, json={"text": text, "session_id": session_id}, timeout=20)
    log(f"[speak] {url1} -> {r1.status_code}")
    if r1.ok:
        return True

    # 2) x-api-key legacy path
    url2 = f"{BASE}/streaming.input"
    hdr2 = {"x-api-key": HEYGEN_API_KEY, "accept": "application/json", "Content-Type": "application/json"}
    tried.append(("x-api-key", url2))
    r2 = requests.post(url2, headers=hdr2, json={"text": text, "session_id": session_id}, timeout=20)
    log(f"[speak] {url2} -> {r2.status_code} body={r2.text[:180]}")
    return r2.ok

# -------------------------------
# UI + sequence
# -------------------------------
st.markdown("### Alessandra • Echo mode (diagnostic)")
with st.spinner("Starting avatar…"):
    session_id, offer_sdp, rtc_config = new_session(AVATAR_ID, VOICE_ID)
    token = create_session_token(session_id)
    st.session_state["HG_TOKEN"] = token
    log(f"[info] session_id={session_id}")
    log("[3] sleeping 1.0s before viewer")
    time.sleep(1.0)

render_viewer_html(token, session_id, offer_sdp, rtc_config, height=600)

# Controls (optional Echo)
if "webrtc_ctx" not in st.session_state:
    st.session_state.webrtc_ctx = None
if "mic_proc" not in st.session_state:
    # only create when user presses Start, to avoid auto mic prompt on load
    st.session_state.mic_proc = None

col1, col2 = st.columns(2, gap="small")
with col1:
    start_clicked = st.button("▶️  Start", type="primary")
with col2:
    stop_clicked = st.button("⏹️  Stop")

if start_clicked and _HAS_WEBRTC:
    if st.session_state.mic_proc is None:
        from streamlit_webrtc import WebRtcMode
        st.session_state.mic_proc = _MicProc()
    st.session_state.webrtc_ctx = webrtc_streamer(
        key="mic-only",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=lambda: st.session_state.mic_proc,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=False,
    )
    st.toast("Mic started.")
    log("[echo] webrtc started")

if stop_clicked:
    ctx = st.session_state.get("webrtc_ctx")
    if ctx and getattr(ctx, "state", None) and ctx.state.playing:
        ctx.stop()
        log("[echo] webrtc stopped")
    st.session_state.webrtc_ctx = None

st.markdown("<div class='caption'>Use Safari/Chrome on iPhone · Allow microphone access when prompted.</div>", unsafe_allow_html=True)

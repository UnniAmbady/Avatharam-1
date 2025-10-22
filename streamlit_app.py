# streamlit_app.py — Stable + verbose diagnostics
# - Prevents repeated session creation (Streamlit re-runs)
# - Adds detailed logs to UI + server
# - Keeps Stage-1 order: new -> create_token -> sleep(1s) -> viewer

import os, json, time, base64
from pathlib import Path
from typing import Optional

import requests
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
    _HAS_WEBRTC = True
except Exception:
    _HAS_WEBRTC = False

st.set_page_config(page_title="Alessandra • Echo (diag)", page_icon="🗣️", layout="centered")
st.markdown("""
<style>
  .block-container { padding-top: .4rem; padding-bottom: 1.2rem; }
  .stButton > button { width:100%; height:56px; font-size:1.05rem; border-radius:12px; }
  .caption { color:#64748b; font-size:.8rem; }
  iframe { border:none; border-radius:14px; }
  /* Hide internal start control of streamlit-webrtc */
  div.st-webrtc > div:has(button) { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Secrets helpers
# ---------------------------
def get_heygen_key() -> Optional[str]:
    s = st.secrets
    for sec, key in [("HeyGen", "heygen_api_key"), ("heygen", "heygen_api_key")]:
        try: return s[sec][key]
        except Exception: pass
    return os.getenv("HEYGEN_API_KEY")

HEYGEN_API_KEY = get_heygen_key()
if not HEYGEN_API_KEY:
    st.error("Missing HeyGen API key. Put in .streamlit/secrets.toml:\n\n[HeyGen]\nheygen_api_key = \"...\"")
    st.stop()

OPENAI_API_KEY = (st.secrets.get("openai", {}) or {}).get("api_key") or os.getenv("OPENAI_API_KEY")

AVATAR_ID = "Alessandra_CasualLook_public"
VOICE_ID  = "0d3f35185d7c4360b9f03312e0264d59"

BASE = "https://api.heygen.com/v1"
API_STREAM_NEW   = f"{BASE}/streaming.new"
API_CREATE_TOKEN = f"{BASE}/streaming.create_token"

HEADERS = {"x-api-key": HEYGEN_API_KEY, "accept": "application/json", "Content-Type": "application/json"}

# ---------------------------
# Diagnostics toggle
# ---------------------------
with st.expander("Diagnostics"):
    VERBOSE = st.checkbox("Verbose logs", value=True)
    st.caption("If it stalls, copy both UI and server logs.")

def log(msg: str):
    if VERBOSE: st.write("▪️", msg)
    print(msg, flush=True)

# ---------------------------
# API helpers
# ---------------------------
def new_session(avatar_id: str, voice_id: Optional[str] = None):
    payload = {"avatar_id": avatar_id}
    if voice_id: payload["voice_id"] = voice_id
    log(f"[1] POST {API_STREAM_NEW}")
    r = requests.post(API_STREAM_NEW, headers=HEADERS, json=payload, timeout=45)
    log(f"[1] status={r.status_code}")
    if not r.ok:
        log(f"[1] body={r.text}")
        r.raise_for_status()
    d = r.json().get("data", {})
    session_id = d.get("session_id")
    offer_sdp  = (d.get("offer") or d.get("sdp") or {}).get("sdp")
    ice2 = d.get("ice_servers2") or []
    ice1 = d.get("ice_servers") or []
    rtc_config = {"iceServers": (ice2 or ice1 or [{"urls": ["stun:stun.l.google.com:19302"]}])}
    if not session_id or not offer_sdp:
        raise RuntimeError("Missing session_id or offer.sdp in streaming.new response")
    return session_id, offer_sdp, rtc_config

def create_session_token(session_id: str) -> str:
    log(f"[2] POST {API_CREATE_TOKEN}")
    r = requests.post(API_CREATE_TOKEN, headers=HEADERS, json={"session_id": session_id}, timeout=45)
    log(f"[2] status={r.status_code}")
    if not r.ok:
        log(f"[2] body={r.text}")
        r.raise_for_status()
    dd = r.json().get("data", {})
    token = dd.get("token") or dd.get("access_token")
    if not token:
        raise RuntimeError("No token/access_token returned by streaming.create_token")
    return token

# ---------------------------
# One-time session guard
# ---------------------------
if "session_ready" not in st.session_state:
    st.session_state.session_ready = False
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "offer_sdp" not in st.session_state:
    st.session_state.offer_sdp = None
if "rtc_config" not in st.session_state:
    st.session_state.rtc_config = None
if "HG_TOKEN" not in st.session_state:
    st.session_state.HG_TOKEN = None

if not st.session_state.session_ready:
    with st.spinner("Starting avatar…"):
        sid, sdp, rtc = new_session(AVATAR_ID, VOICE_ID)
        tok = create_session_token(sid)
        st.session_state.session_id = sid
        st.session_state.offer_sdp  = sdp
        st.session_state.rtc_config = rtc
        st.session_state.HG_TOKEN   = tok
        log(f"[info] session_id={sid}")
        log("[3] sleeping 1.0s before viewer")
        time.sleep(1.0)
        st.session_state.session_ready = True

# ---------------------------
# Render viewer (base64 SDP to avoid escapes)
# ---------------------------
def render_viewer_html(token: str, session_id: str, offer_sdp: str, rtc_config: dict, height: int = 600):
    viewer_path = Path(__file__).parent / "viewer.html"
    if not viewer_path.exists():
        st.error("viewer.html not found next to streamlit_app.py.")
        st.stop()
    sdp_b64 = base64.b64encode(offer_sdp.encode("utf-8")).decode("ascii")
    raw = viewer_path.read_text(encoding="utf-8")
    html = (raw
        .replace("__SESSION_TOKEN__", token)
        .replace("__SESSION_ID__", session_id)
        .replace("__AVATAR_NAME__", "Alessandra Casual Look")
        .replace("__OFFER_SDP_B64__", sdp_b64)
        .replace("__RTC_CONFIG__", json.dumps(rtc_config))
        .replace("__API_BASE__", BASE)
    )
    components.html(html, height=600, scrolling=False)

render_viewer_html(
    st.session_state.HG_TOKEN,
    st.session_state.session_id,
    st.session_state.offer_sdp,
    st.session_state.rtc_config,
    height=600,
)

# ---------------------------
# Optional: mic echo
# ---------------------------
class _MicProc(AudioProcessorBase):
    def __init__(self): self.buf=b""; self.sample_rate=16000
    def recv_audio(self, frame):
        pcm16 = frame.to_ndarray(format="s16")
        if pcm16.ndim==2 and pcm16.shape[0] > 1: pcm16=pcm16[0:1,:]
        pcm16 = np.squeeze(pcm16)
        in_rate = frame.sample_rate
        if in_rate != self.sample_rate:
            import numpy as _np
            dur = pcm16.shape[0]/in_rate
            new_len = int(dur*self.sample_rate)
            pcm16 = _np.interp(_np.linspace(0, pcm16.shape[0], new_len, endpoint=False),
                               _np.arange(pcm16.shape[0]), pcm16.astype(_np.float32)).astype(np.int16)
        self.buf += pcm16.tobytes()
        return frame

def speak_text(session_id: str, text: str):
    # Try both common endpoints; log both results.
    url1 = f"{BASE}/streaming/session/{session_id}/speak"
    hdr1 = {"Authorization": f"Bearer {st.session_state.get('HG_TOKEN','')}",
            "accept": "application/json", "Content-Type": "application/json"}
    r1 = requests.post(url1, headers=hdr1, json={"text": text, "session_id": session_id}, timeout=20)
    log(f"[speak] {url1} -> {r1.status_code} {r1.text[:180]}")
    if r1.ok: return True

    url2 = f"{BASE}/streaming.input"
    hdr2 = {"x-api-key": HEYGEN_API_KEY, "accept": "application/json", "Content-Type": "application/json"}
    r2 = requests.post(url2, headers=hdr2, json={"text": text, "session_id": session_id}, timeout=20)
    log(f"[speak] {url2} -> {r2.status_code} {r2.text[:180]}")
    return r2.ok

if "webrtc_ctx" not in st.session_state: st.session_state.webrtc_ctx = None
if "mic_proc" not in st.session_state:   st.session_state.mic_proc = None

col1, col2 = st.columns(2, gap="small")
with col1:
    start_clicked = st.button("▶️  Start", type="primary")
with col2:
    stop_clicked  = st.button("⏹️  Stop")

if start_clicked and _HAS_WEBRTC:
    if st.session_state.mic_proc is None:
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

st.markdown("<div class='caption'>Use Safari/Chrome on iPhone • Allow microphone access when prompted.</div>", unsafe_allow_html=True)

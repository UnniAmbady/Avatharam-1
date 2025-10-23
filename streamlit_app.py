# streamlit_app.py — Step-by-step control bar + phase-driven viewer
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

st.set_page_config(page_title="Alessandra • Step Runner", page_icon="🗣️", layout="centered")

st.markdown("""
<style>
  .block-container { padding-top:.4rem; padding-bottom:1rem; }
  /* compact left-aligned control bar */
  .ctrl-row { display:flex; flex-wrap:wrap; gap:6px; }
  .ctrl-row .stButton > button { padding:6px 10px; height:38px; font-size:.85rem; border-radius:10px; }
  .stButton > button[kind=primary] { background:#ef4444; }
  iframe { border:none; border-radius:14px; }
  /* hide internal start of streamlit-webrtc (prevents flashing red popup) */
  div.st-webrtc > div:has(button) { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Secrets
# ---------------------------
def get_heygen_key() -> Optional[str]:
    s = st.secrets
    for sec,k in [("HeyGen","heygen_api_key"),("heygen","heygen_api_key")]:
        try: return s[sec][k]
        except Exception: pass
    return os.getenv("HEYGEN_API_KEY")

HEYGEN_API_KEY = get_heygen_key()
if not HEYGEN_API_KEY:
    st.error("Missing HeyGen API key in .streamlit/secrets.toml under [HeyGen]."); st.stop()

OPENAI_API_KEY = (st.secrets.get("openai", {}) or {}).get("api_key") or os.getenv("OPENAI_API_KEY")

AVATAR_ID = "Alessandra_CasualLook_public"
VOICE_ID  = "0d3f35185d7c4360b9f03312e0264d59"

BASE = "https://api.heygen.com/v1"
API_STREAM_NEW   = f"{BASE}/streaming.new"
API_CREATE_TOKEN = f"{BASE}/streaming.create_token"
HEADERS = {"x-api-key": HEYGEN_API_KEY, "accept":"application/json", "Content-Type":"application/json"}

# ---------------------------
# Session storage
# ---------------------------
if "sid" not in st.session_state: st.session_state.sid = None
if "offer_sdp" not in st.session_state: st.session_state.offer_sdp = None
if "rtc" not in st.session_state: st.session_state.rtc = None
if "token" not in st.session_state: st.session_state.token = None
if "phase" not in st.session_state: st.session_state.phase = "idle"  # idle|new|token|render|sdp_bearer|sdp_plain|sdp_nocors|mic_start|mic_stop

# ---------------------------
# API helpers
# ---------------------------
def new_session(avatar_id: str, voice_id: Optional[str] = None):
    payload = {"avatar_id": avatar_id}
    if voice_id: payload["voice_id"] = voice_id
    r = requests.post(API_STREAM_NEW, headers=HEADERS, json=payload, timeout=45)
    r.raise_for_status()
    d = r.json()["data"]
    sid = d["session_id"]
    offer_sdp = (d.get("offer") or d.get("sdp") or {})["sdp"]
    ice2 = d.get("ice_servers2") or []
    ice1 = d.get("ice_servers") or []
    rtc = {"iceServers": (ice2 or ice1 or [{"urls":["stun:stun.l.google.com:19302"]}])}
    return sid, offer_sdp, rtc

def create_session_token(session_id: str) -> str:
    r = requests.post(API_CREATE_TOKEN, headers=HEADERS, json={"session_id": session_id}, timeout=45)
    r.raise_for_status()
    d = r.json()["data"]
    return d.get("token") or d.get("access_token")

# ---------------------------
# Control bar (single, left-aligned row)
# ---------------------------
st.markdown("##### Controls")
col = st.container()
with col:
    c = st.columns(8, gap="small")
    if c[0].button("1) New"):
        st.session_state.sid, st.session_state.offer_sdp, st.session_state.rtc = new_session(AVATAR_ID, VOICE_ID)
        st.session_state.phase = "new"
    if c[1].button("2) Token"):
        if not st.session_state.sid: st.warning("Run 1) New first.")
        else:
            st.session_state.token = create_session_token(st.session_state.sid)
            st.session_state.phase = "token"
    if c[2].button("3) Render"):
        if not (st.session_state.sid and st.session_state.token): st.warning("Run 1) & 2) first.")
        else:
            st.session_state.phase = "render"
    if c[3].button("4A SDP Bearer"):
        st.session_state.phase = "sdp_bearer"
    if c[4].button("4B SDP Plain"):
        st.session_state.phase = "sdp_plain"
    if c[5].button("4C SDP no-cors"):
        st.session_state.phase = "sdp_nocors"
    if c[6].button("5) Mic ▶"):
        st.session_state.phase = "mic_start"
    if c[7].button("⏹ Mic"):
        st.session_state.phase = "mic_stop"

# ---------------------------
# Viewer render (phase-driven)
# ---------------------------
def render_viewer():
    viewer_path = Path(__file__).parent / "viewer.html"
    if not viewer_path.exists():
        st.error("viewer.html not found next to streamlit_app.py."); st.stop()
    sdp_b64 = base64.b64encode((st.session_state.offer_sdp or "").encode("utf-8")).decode("ascii") if st.session_state.offer_sdp else ""
    rtc_literal = json.dumps(st.session_state.rtc or {})
    raw = viewer_path.read_text(encoding="utf-8")
    html = (raw
        .replace("__SESSION_TOKEN__", st.session_state.token or "")
        .replace("__SESSION_ID__", st.session_state.sid or "")
        .replace("__OFFER_SDP_B64__", sdp_b64)
        .replace("__RTC_CONFIG_LITERAL__", rtc_literal)
        .replace("__API_BASE__", BASE)
        .replace("__PHASE__", st.session_state.phase)  # tells viewer what to do
    )
    components.html(html, height=620, scrolling=False)

# auto-render viewer once you’ve created session+token (or any time you press Render / SDP buttons)
if st.session_state.phase in {"render","sdp_bearer","sdp_plain","sdp_nocors"} and st.session_state.sid and st.session_state.token:
    render_viewer()
elif st.session_state.phase in {"new","token"}:
    # show a small hint to press Render next
    st.info("Press **3) Render** to load viewer. Then try **4A/4B/4C** for SDP submit.")

# ---------------------------
# Optional mic echo (separate from viewer)
# ---------------------------
class _MicProc(AudioProcessorBase):
    def __init__(self): self.buf=b""; self.sample_rate=16000
    def recv_audio(self, frame):
        pcm16 = frame.to_ndarray(format="s16")
        if pcm16.ndim==2 and pcm16.shape[0]>1: pcm16=pcm16[0:1,:]
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

if "webrtc_ctx" not in st.session_state: st.session_state.webrtc_ctx = None
if "mic_proc" not in st.session_state:   st.session_state.mic_proc = None

if st.session_state.phase == "mic_start" and _HAS_WEBRTC:
    if st.session_state.mic_proc is None: st.session_state.mic_proc = _MicProc()
    st.session_state.webrtc_ctx = webrtc_streamer(
        key="mic-only",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=lambda: st.session_state.mic_proc,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=False,
    )
    st.toast("Mic started.")

if st.session_state.phase == "mic_stop":
    ctx = st.session_state.get("webrtc_ctx")
    if ctx and getattr(ctx, "state", None) and ctx.state.playing: ctx.stop()
    st.session_state.webrtc_ctx = None
    st.toast("Mic stopped.")

st.caption("Use Safari/Chrome on iPhone • Slide up the Diagnostics bar in the video to see logs.")

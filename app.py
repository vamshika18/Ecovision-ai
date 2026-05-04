# ============================================================
# app.py  –  EcoVision AI — Smart Waste Classification Dashboard
# Live WebRTC webcam stream + all features | Dark premium theme
# ============================================================

import streamlit as st
from ultralytics import YOLO
from waste_analysis import analyze_waste
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import io
import csv
import av
import threading
from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="EcoVision AI — Smart Waste Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:#0a0d0f; --surface:#111518; --surface2:#181d22; --border:#1f2a2e;
  --green:#00e676; --green-dim:#00e67222; --amber:#ffab00; --amber-dim:#ffab0022;
  --red:#ff1744; --red-dim:#ff174422; --cyan:#18ffff; --cyan-dim:#18ffff18;
  --text:#e8f0f2; --muted:#6b8a92; --radius:12px;
}
html, body, [data-testid="stApp"] {
  background:var(--bg) !important; color:var(--text) !important;
  font-family:'DM Sans',sans-serif !important;
}
#MainMenu, footer, header { visibility:hidden; }
[data-testid="stDecoration"],[data-testid="stToolbar"] { display:none; }

[data-testid="stSidebar"] {
  background:var(--surface) !important;
  border-right:1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color:var(--text) !important; }

[data-testid="stTabs"] button {
  background:var(--surface2) !important; color:var(--muted) !important;
  border:1px solid var(--border) !important; border-radius:8px 8px 0 0 !important;
  font-family:'Space Mono',monospace !important; font-size:0.76rem !important;
  letter-spacing:0.05em !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  background:var(--green-dim) !important; color:var(--green) !important;
  border-bottom-color:var(--green) !important;
}
[data-testid="stMetric"] {
  background:var(--surface2) !important; border:1px solid var(--border) !important;
  border-radius:var(--radius) !important; padding:16px !important;
}
[data-testid="stMetricValue"] {
  font-family:'Space Mono',monospace !important; color:var(--green) !important;
}
[data-testid="stMetricLabel"] { color:var(--muted) !important; font-size:0.75rem !important; }
[data-testid="stButton"] > button {
  background:var(--green-dim) !important; color:var(--green) !important;
  border:1px solid var(--green) !important; border-radius:8px !important;
  font-family:'Space Mono',monospace !important; font-size:0.76rem !important;
}
[data-testid="stButton"] > button:hover { background:var(--green) !important; color:#000 !important; }
[data-testid="stFileUploader"] {
  background:var(--surface2) !important; border:1px dashed var(--border) !important;
  border-radius:var(--radius) !important;
}
::-webkit-scrollbar { width:5px; background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:10px; }

/* ── PIN sidebar — never collapse ── */
[data-testid="stSidebar"] {
  min-width: 240px !important;
  max-width: 260px !important;
  display: flex !important;
  visibility: visible !important;
  transform: translateX(0px) !important;
  transition: none !important;
}
/* Hide ALL collapse/expand controls */
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="collapsedControl"]         { display: none !important; }
[data-testid="stSidebarNavCollapseIcon"]  { display: none !important; }
button[aria-label="Collapse sidebar"]    { display: none !important; }
button[aria-label="Expand sidebar"]      { display: none !important; }

/* Keep main content always offset from sidebar */
[data-testid="stAppViewContainer"] > section:first-of-type {
  margin-left: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── HTML helpers ──────────────────────────────────────────────
def status_badge(status, risk):
    colors = {
        "Clean Recyclable Waste":     ("#00e676","#00e67218","✅"),
        "Contaminated / Mixed Waste": ("#ffab00","#ffab0018","⚠️"),
        "Non-Recyclable Waste":       ("#ff1744","#ff174418","❌"),
        "No Waste Detected":          ("#6b8a92","#6b8a9215","🔍"),
    }
    c,bg,icon = colors.get(status,("#6b8a92","#6b8a9215","•"))
    return f"""<div style="background:{bg};border:1px solid {c};border-radius:10px;
        padding:14px 20px;margin:10px 0;display:flex;align-items:center;gap:12px;">
      <span style="font-size:1.5rem">{icon}</span>
      <div>
        <div style="color:{c};font-family:'Space Mono',monospace;font-size:0.95rem;font-weight:700">{status}</div>
        <div style="color:#6b8a92;font-size:0.76rem;margin-top:3px">Risk Level: <b style="color:{c}">{risk}</b></div>
      </div></div>"""

def section_header(title, sub="MODULE"):
    return f"""<div style="margin:20px 0 10px">
      <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#00e676;
                  letter-spacing:.15em;text-transform:uppercase;margin-bottom:4px">◆ {sub}</div>
      <div style="font-size:1.15rem;font-weight:600;color:#e8f0f2">{title}</div>
      <div style="width:36px;height:2px;background:linear-gradient(90deg,#00e676,transparent);margin-top:5px"></div>
    </div>"""

def info_box(text, kind="info"):
    cfg = {"success":("#00e676","#00e67215"),"warning":("#ffab00","#ffab0015"),
           "error":("#ff1744","#ff174415"),"info":("#18ffff","#18ffff10")}
    c,bg = cfg.get(kind,cfg["info"])
    return f'<div style="background:{bg};border-left:3px solid {c};border-radius:0 8px 8px 0;padding:11px 15px;margin:7px 0;font-size:0.85rem;color:{c}">{text}</div>'

def pill_row(r, nr, cp):
    def pill(lbl,val,color):
        return f'<div style="background:{color}15;border:1px solid {color}40;border-radius:10px;padding:12px;text-align:center;flex:1"><div style="color:{color};font-family:Space Mono,monospace;font-size:1.3rem;font-weight:700">{val}</div><div style="color:#6b8a92;font-size:0.7rem;margin-top:3px;text-transform:uppercase">{lbl}</div></div>'
    return f'<div style="display:flex;gap:10px;margin:12px 0">{pill("Recyclable",r,"#00e676")}{pill("Non-Recyclable",nr,"#ff1744")}{pill("Contamination",str(cp)+"%","#ffab00")}</div>'

def contam_alert_html():
    return """<div style="background:#ff174418;border:2px solid #ff1744;border-radius:10px;
        padding:13px 20px;text-align:center;color:#ff1744;font-family:'Space Mono',monospace;
        font-size:0.9rem;letter-spacing:.05em;margin:10px 0">
        ⚠️  HIGH CONTAMINATION ALERT — exceeds 60%</div>"""

# ── Session state ─────────────────────────────────────────────
for k,v in [("history",[]),("total_r",0),("total_n",0)]:
    if k not in st.session_state: st.session_state[k] = v

# ── Model (cached) ────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

RECYCLABLE = {"can","cardboard_bowl","cardboard_box","plastic_bag","plastic_bottle",
              "plastic_bottle_cap","plastic_box","plastic_cutlery","plastic_cup",
              "plastic_cup_lid","reusable_paper","scrap_paper","scrap_plastic","paper","metal_can"}

# ── Config ────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45  # slightly lower = catches objects faster
SMOOTHING_FRAMES     = 3   # 3 frame window
MIN_AGREE_FRAMES     = 1   # 1 confident detection = lock instantly

# ── RTC config (STUN for network traversal) ───────────────────
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ══════════════════════════════════════════════════════════════
# LIVE VIDEO PROCESSOR — runs in background thread
# ══════════════════════════════════════════════════════════════
class WasteDetector(VideoProcessorBase):
    """
    Snap-and-freeze detector:
    - Scans frames looking for a detection
    - Once object is detected consistently → FREEZES on that frame
    - Shows frozen annotated frame until user clicks Reset
    - Much faster: only runs YOLO every 3rd frame while scanning
    """
    def __init__(self):
        self._model       = model
        self._history     = deque(maxlen=SMOOTHING_FRAMES)
        self._lock        = threading.Lock()
        self.analysis     = analyze_waste([], [])
        self.is_stable    = False
        self.frozen       = False          # True = snap taken, showing frozen frame
        self.frozen_frame = None           # The frozen annotated frame
        self._frame_skip  = 0             # Process every 3rd frame only
        self.reset        = False          # Signal from UI to unfreeze

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        import cv2
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        # ── Convert frame ─────────────────────────────────────
        img = frame.to_ndarray(format="bgr24")
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # ── If frozen: return frozen frame immediately ─────────
        with self._lock:
            if self.frozen and self.frozen_frame is not None:
                if self.reset:
                    self.frozen       = False
                    self.frozen_frame = None
                    self.is_stable    = False
                    self.analysis     = analyze_waste([], [])
                    self._history.clear()
                    self.reset        = False
                else:
                    return av.VideoFrame.from_ndarray(self.frozen_frame, format="bgr24")

        # ── YOLO on every frame at half resolution (fast) ────
        results = self._model(img, verbose=False, conf=CONFIDENCE_THRESHOLD, imgsz=320)
        frame_classes, frame_confs = [], []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf >= CONFIDENCE_THRESHOLD:
                frame_classes.append(self._model.names[int(box.cls[0])])
                frame_confs.append(conf)

        with self._lock:
            self._history.append((frame_classes, frame_confs))
            all_cls, conf_map = [], {}
            for fc, fconf in self._history:
                for c, cf in zip(fc, fconf):
                    all_cls.append(c)
                    conf_map.setdefault(c, []).append(cf)
            counts      = Counter(all_cls)
            stable_cls  = [c for c,n in counts.items() if n >= MIN_AGREE_FRAMES]
            stable_conf = [sum(conf_map[c])/len(conf_map[c]) for c in stable_cls]
            if stable_cls:
                self.analysis  = analyze_waste(stable_cls, stable_conf)
                self.is_stable = True
            else:
                self.analysis  = analyze_waste([], [])
                self.is_stable = False

        # Annotated frame with bounding boxes
        annotated = results[0].plot()

        # ── Always draw HUD on every frame ────────────────────
        h, w = annotated.shape[:2]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0,0), (w, 155), (12,15,18), -1)
        cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0, annotated)

        ana   = self.analysis
        dot_c = (0,230,118) if self.is_stable else (0,140,255)
        label = "LOCKED" if self.is_stable else "SCANNING..."
        cv2.circle(annotated, (w-18, 18), 7, dot_c, -1)
        cv2.putText(annotated, label, (w-100, 23), FONT, 0.40, dot_c, 1, cv2.LINE_AA)

        # Show scanning progress bar (how many frames collected out of needed)
        if not self.is_stable:
            frames_collected = len(self._history)
            bar_w = int((frames_collected / SMOOTHING_FRAMES) * 200)
            cv2.rectangle(annotated, (w-220, h-18), (w-20, h-8), (30,40,35), -1)
            cv2.rectangle(annotated, (w-220, h-18), (w-220+bar_w, h-8), (0,180,80), -1)
            cv2.putText(annotated, "DETECTING...", (w-220, h-22),
                        FONT, 0.38, (0,180,80), 1, cv2.LINE_AA)

        sc = {"Clean Recyclable Waste":(0,230,118),
              "Contaminated / Mixed Waste":(0,130,255),
              "Non-Recyclable Waste":(30,30,220)}.get(ana["status"],(180,180,180))
        risk_c = {"Clean":(0,230,118),"High":(30,30,220),
                  "Medium":(0,165,255),"Low":(0,230,230),"None":(180,180,180)}.get(
                  ana["risk_level"],(255,255,255))

        lines = [
            (f"Status       : {ana['status']}",                                          (10,30),  sc,            0.58),
            (f"Risk Level   : {ana['risk_level']}",                                      (10,58),  risk_c,        0.58),
            (f"Contamination: {ana['contamination_percent']}%",                          (10,86),  (0,200,255),   0.58),
            (f"Recyclable: {ana['recyclable_count']}   Non-Recyclable: {ana['non_recyclable_count']}",
                                                                                         (10,114), (200,200,200), 0.52),
            (f"Tip: {ana['recommendation'][:62]}",                                       (10,142), (255,200,50),  0.46),
        ]
        for text, pos, color, scale in lines:
            cv2.putText(annotated, text, pos, FONT, scale, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(annotated, text, pos, FONT, scale, color,   1, cv2.LINE_AA)

        # ── FREEZE on stable detection ────────────────────────
        with self._lock:
            if self.is_stable and not self.frozen:
                ana = self.analysis
                detected = ", ".join(ana.get("recyclable_items",[]) + ana.get("non_recyclable_items",[]))
                banner = f"LOCKED: {detected[:55]}  |  Click Reset to scan again" if detected else "OBJECT DETECTED - FROZEN  |  Click Reset to scan again"
                # Draw green freeze banner at bottom
                cv2.rectangle(annotated, (0, h-42), (w, h), (0,25,5), -1)
                cv2.rectangle(annotated, (0, h-42), (w, h-41), (0,230,118), -1)  # top border line
                cv2.putText(annotated, banner,
                            (10, h-14), FONT, 0.46, (0,230,118), 1, cv2.LINE_AA)
                self.frozen_frame = annotated.copy()
                self.frozen       = True

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ── Detection helpers ─────────────────────────────────────────
def run_detection(image_np):
    # Convert RGBA (4-channel PNG) or grayscale to RGB (3-channel)
    if image_np.shape[-1] == 4:
        image_np = image_np[:, :, :3]          # drop alpha channel
    elif len(image_np.shape) == 2:
        image_np = np.stack([image_np]*3, axis=-1)  # grayscale to RGB
    res = model(image_np)
    cls_list, conf_list = [], []
    for box in res[0].boxes:
        cls_list.append(model.names[int(box.cls[0])])
        conf_list.append(float(box.conf[0]))
    analysis = analyze_waste(cls_list, conf_list)
    return res[0].plot(), analysis, cls_list, conf_list

def save_history(analysis, classes, source):
    st.session_state.history.append({
        "timestamp":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":          source,
        "status":          analysis["status"],
        "risk_level":      analysis["risk_level"],
        "contamination_%": analysis["contamination_percent"],
        "recyclable":      analysis["recyclable_count"],
        "non_recyclable":  analysis["non_recyclable_count"],
        "items":           ", ".join(classes) if classes else "None",
    })
    st.session_state.total_r += analysis["recyclable_count"]
    st.session_state.total_n += analysis["non_recyclable_count"]

def show_result(annotated, analysis, classes, confs):
    c1, c2 = st.columns([1.3,1], gap="large")
    with c1:
        st.image(annotated, caption="Detection Output", use_container_width=True)
    with c2:
        st.markdown(section_header("Analysis Report","RESULT"), unsafe_allow_html=True)
        st.markdown(status_badge(analysis["status"], analysis["risk_level"]), unsafe_allow_html=True)
        st.markdown(pill_row(analysis["recyclable_count"],
                             analysis["non_recyclable_count"],
                             analysis["contamination_percent"]), unsafe_allow_html=True)
        kind = {"Clean Recyclable Waste":"success","Contaminated / Mixed Waste":"warning",
                "Non-Recyclable Waste":"error"}.get(analysis["status"],"info")
        st.markdown(info_box(analysis["alert"], kind), unsafe_allow_html=True)
        st.markdown(info_box("💡 "+analysis["recommendation"],"info"), unsafe_allow_html=True)
        st.markdown(f'<div style="color:#6b8a92;font-size:0.78rem;margin-top:8px;font-style:italic">📖 {analysis["insight"]}</div>', unsafe_allow_html=True)
        if classes:
            st.markdown(section_header("Detected Items","OBJECTS"), unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Item":       classes,
                "Confidence": [f"{c:.0%}" for c in confs],
                "Category":   ["♻️ Recyclable" if c in RECYCLABLE else "🚫 Non-Recyclable" for c in classes]
            }), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:18px 0 8px;text-align:center">
      <div style="font-size:2.2rem">♻️</div>
      <div style="font-family:'Space Mono',monospace;color:#00e676;font-size:0.9rem;font-weight:700;letter-spacing:.1em;margin-top:6px">ECOVISION AI</div>
      <div style="color:#6b8a92;font-size:0.7rem;margin-top:3px">Smart Waste Classification</div>
    </div>
    <hr style="border-color:#1f2a2e;margin:8px 0 14px">
    """, unsafe_allow_html=True)

    total_scans = len(st.session_state.history)
    avg_c = (sum(h["contamination_%"] for h in st.session_state.history)/total_scans) if total_scans else 0

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px">
      {"".join([f'<div style="background:#111518;border:1px solid #1f2a2e;border-radius:8px;padding:10px;text-align:center"><div style="color:{c};font-family:Space Mono,monospace;font-size:1.3rem">{v}</div><div style="color:#6b8a92;font-size:0.66rem;margin-top:2px">{l}</div></div>'
      for v,l,c in [(total_scans,"TOTAL SCANS","#00e676"),(f"{avg_c:.0f}%","AVG CONTAM.","#ffab00"),
                    (st.session_state.total_r,"RECYCLABLE","#00e676"),(st.session_state.total_n,"NON-RECYCLE","#ff1744")]])}
    </div>
    <hr style="border-color:#1f2a2e;margin:8px 0 14px">
    <div style="font-size:0.8rem;line-height:2.1;color:#aaa">
      🧠 &nbsp;YOLOv8 Custom Trained<br>
      🏷️ &nbsp;{len(model.names)} Waste Classes<br>
      📊 &nbsp;mAP50: <span style="color:#00e676">0.914</span><br>
      🎯 &nbsp;Conf. Threshold: 0.50<br>
      🎥 &nbsp;Live Stream: <span style="color:#00e676">WebRTC</span><br>
      🟢 &nbsp;<span style="color:#00e676">LIVE</span>
    </div>
    <hr style="border-color:#1f2a2e;margin:14px 0">
    """, unsafe_allow_html=True)

    # Export CSV — always visible, disabled when empty
    if st.session_state.history:
        buf = io.StringIO()
        w   = csv.DictWriter(buf, fieldnames=list(st.session_state.history[0].keys()))
        w.writeheader(); w.writerows(st.session_state.history)
        csv_data = buf.getvalue()
        st.download_button("📥 Export CSV Report", csv_data,
                           file_name=f"waste_report_{datetime.date.today()}.csv",
                           mime="text/csv", use_container_width=True)
    else:
        st.markdown("""
        <div style="background:#111518;border:1px solid #1f2a2e;border-radius:8px;
                    padding:10px;text-align:center;color:#6b8a92;font-size:0.78rem;margin-bottom:4px">
          📥 Export CSV<br><span style="font-size:0.7rem">Available after first scan</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Session", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_r = st.session_state.total_n = 0
        st.rerun()

# ─────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 8px;border-bottom:1px solid #1f2a2e;margin-bottom:20px">
  <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#00e676;
              letter-spacing:.2em;text-transform:uppercase;margin-bottom:8px">◆ AI-POWERED COMPUTER VISION</div>
  <div style="font-size:1.9rem;font-weight:600;color:#e8f0f2;line-height:1.25">
    Smart Waste Classification<br><span style="color:#00e676">&amp; Recycling Optimization</span>
  </div>
  <div style="color:#6b8a92;font-size:0.85rem;margin-top:8px">
    Live webcam stream · Real-time detection · Contamination analysis · Recycling guidance
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎥  Live Webcam", "🖼️  Image Upload", "📦  Batch Detection",
    "📊  Analytics",   "🧠  AI Tips"
])

# ══ TAB 1 — LIVE WEBCAM (WebRTC) ══════════════════════════════
with tab1:
    st.markdown(section_header("Live Webcam Detection","REAL-TIME STREAM"), unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#00e67210;border:1px solid #00e67235;border-radius:10px;
                padding:12px 18px;margin-bottom:16px;font-size:0.84rem;color:#00e676">
      🎥 &nbsp;<b>True live stream</b> — continuous frames, same engine as realtime.py,
      running directly in your browser via WebRTC.
      &nbsp;|&nbsp; 🟢 STABLE indicator confirms locked detection.
    </div>
    """, unsafe_allow_html=True)

    col_stream, col_live = st.columns([1.4, 1], gap="large")

    with col_stream:
        # Launch WebRTC streamer — this is the live webcam
        ctx = webrtc_streamer(
            key="waste-detector",
            video_processor_factory=WasteDetector,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
            async_processing=True,
        )

    with col_live:
        st.markdown("""
        <div style="background:#111518;border:1px solid #1f2a2e;border-radius:12px;padding:18px;margin-top:6px">
          <div style="font-family:'Space Mono',monospace;color:#00e676;font-size:0.68rem;
                      letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px">LIVE ANALYSIS PANEL</div>
          <div style="color:#6b8a92;font-size:0.8rem;margin-bottom:14px">
            Results update live on the video feed.<br>Watch for the <b style="color:#00e676">🟢 STABLE</b> indicator.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Live result display
        if ctx.video_processor:
            proc = ctx.video_processor
            ana  = proc.analysis

            # ── Auto-rerun when frozen to update right panel ─
            if proc.frozen and ana["status"] != "No Waste Detected":
                st.markdown("""
                <div style="background:#00e67220;border:1px solid #00e676;border-radius:8px;
                            padding:10px 14px;text-align:center;margin-bottom:8px">
                  <span style="color:#00e676;font-family:Space Mono,monospace;font-size:0.82rem;font-weight:700">
                    🔒 OBJECT CAPTURED — Frame Frozen
                  </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#ffab0012;border:1px solid #ffab0050;border-radius:8px;
                            padding:10px 14px;text-align:center;margin-bottom:8px">
                  <span style="color:#ffab00;font-family:Space Mono,monospace;font-size:0.82rem">
                    🔍 SCANNING — Hold item 30–60 cm from camera
                  </span>
                </div>""", unsafe_allow_html=True)

            # ── Analysis result ───────────────────────────────
            st.markdown(
                status_badge(ana["status"], ana["risk_level"]) +
                pill_row(ana["recyclable_count"], ana["non_recyclable_count"], ana["contamination_percent"]) +
                info_box(ana["alert"], {"Clean Recyclable Waste":"success",
                                        "Contaminated / Mixed Waste":"warning",
                                        "Non-Recyclable Waste":"error"}.get(ana["status"],"info")) +
                info_box("💡 " + ana["recommendation"], "info") +
                f'<div style="color:#6b8a92;font-size:0.77rem;margin-top:8px;font-style:italic">📖 {ana["insight"]}</div>',
                unsafe_allow_html=True
            )

            if ana["contamination_percent"] >= 60:
                st.markdown(contam_alert_html(), unsafe_allow_html=True)

            # ── Buttons ───────────────────────────────────────
            if st.button("🔄 Reset & Scan Again", use_container_width=True):
                proc.reset = True
                st.rerun()

            # Auto-rerun every 1s while scanning to keep panel fresh
            if not proc.frozen:
                import time
                time.sleep(0.8)
                st.rerun()

            # Save to history button
            if st.button("💾 Save Current Detection", use_container_width=True):
                if ana["status"] != "No Waste Detected":
                    save_history(ana, ana.get("recyclable_items",[]) + ana.get("non_recyclable_items",[]), "webcam")
                    st.markdown("""
                    <div style="background:#00e67215;border:1px solid #00e67240;border-radius:8px;
                                padding:12px 16px;margin-top:8px;text-align:center">
                      <div style="color:#00e676;font-size:0.88rem;font-weight:600">✅ Saved to session history!</div>
                      <div style="color:#6b8a92;font-size:0.78rem;margin-top:5px">
                        👉 Click the <b style="color:#00e676">📊 Analytics</b> tab above to view all detections,
                        charts &amp; export your CSV report.
                      </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.warning("Nothing detected to save. Point camera at a waste item first.")
        else:
            st.markdown("""
            <div style="background:#111518;border:1px dashed #1f2a2e;border-radius:10px;
                        padding:32px;text-align:center;margin-top:10px">
              <div style="font-size:2rem;margin-bottom:10px">🎥</div>
              <div style="color:#6b8a92;font-size:0.85rem">
                Click <b style="color:#00e676">START</b> to begin live detection.<br>
                Allow camera access when prompted.
              </div>
            </div>""", unsafe_allow_html=True)

    # How-to guide
    st.markdown(section_header("Tips for Best Detection","GUIDE"), unsafe_allow_html=True)
    g1, g2, g3, g4 = st.columns(4)
    for col, icon, title, tip in zip(
        [g1,g2,g3,g4],
        ["💡","📐","🖐️","⏱️"],
        ["Good Lighting","Right Distance","Keep Still","Wait for STABLE"],
        ["Use bright natural or indoor light — avoid backlighting",
         "Hold item 30–60 cm away, fully in frame",
         "Keep item still — wait for 🟢 STABLE indicator",
         "Detection stabilises after ~8 frames automatically"]
    ):
        col.markdown(f"""
        <div style="background:#111518;border:1px solid #1f2a2e;border-radius:10px;
                    padding:16px;text-align:center">
          <div style="font-size:1.6rem;margin-bottom:8px">{icon}</div>
          <div style="color:#00e676;font-family:Space Mono,monospace;font-size:0.72rem;font-weight:700;
                      margin-bottom:6px;letter-spacing:.05em">{title}</div>
          <div style="color:#aaa;font-size:0.78rem;line-height:1.6">{tip}</div>
        </div>""", unsafe_allow_html=True)

# ══ TAB 2 — IMAGE UPLOAD ══════════════════════════════════════
with tab2:
    st.markdown(section_header("Image Upload Detection","SINGLE IMAGE"), unsafe_allow_html=True)
    up = st.file_uploader("Upload a waste image (JPG / PNG)", type=["jpg","jpeg","png"])
    if up:
        img = np.array(Image.open(up).convert('RGB'))
        ca, cb = st.columns(2, gap="large")
        with ca:
            st.markdown('<div style="color:#6b8a92;font-size:0.73rem;margin-bottom:5px">ORIGINAL IMAGE</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
        with st.spinner("🔍 Analysing..."):
            ann, ana, cls, cfs = run_detection(img)
        with cb:
            st.markdown('<div style="color:#00e676;font-size:0.73rem;margin-bottom:5px">DETECTION OUTPUT</div>', unsafe_allow_html=True)
            st.image(ann, use_container_width=True)
        save_history(ana, cls, "upload")
        if ana["contamination_percent"] >= 60:
            st.markdown(contam_alert_html(), unsafe_allow_html=True)
        show_result(ann, ana, cls, cfs)

# ══ TAB 3 — BATCH DETECTION ═══════════════════════════════════
with tab3:
    st.markdown(section_header("Batch Detection","MULTIPLE IMAGES"), unsafe_allow_html=True)
    st.markdown('<div style="color:#6b8a92;font-size:0.83rem;margin-bottom:14px">Upload up to 10 images — all classified and added to session analytics.</div>', unsafe_allow_html=True)
    files = st.file_uploader("Upload multiple images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if files:
        if len(files) > 10:
            st.warning("Max 10 — processing first 10.")
            files = files[:10]
        prog    = st.progress(0, text="Processing...")
        results = []
        for i, f in enumerate(files):
            ann, ana, cls, cfs = run_detection(np.array(Image.open(f).convert('RGB')))
            save_history(ana, cls, "batch")
            results.append({"name":f.name,"ann":ann,"ana":ana})
            prog.progress((i+1)/len(files), text=f"Processing {f.name}...")
        prog.empty()

        clean  = sum(1 for r in results if r["ana"]["status"]=="Clean Recyclable Waste")
        contam = sum(1 for r in results if r["ana"]["status"]=="Contaminated / Mixed Waste")
        nonrec = sum(1 for r in results if r["ana"]["status"]=="Non-Recyclable Waste")

        st.markdown(f"""
        <div style="display:flex;gap:10px;margin:14px 0;flex-wrap:wrap">
          {"".join([f'<div style="background:{bg};border:1px solid {bc};border-radius:10px;padding:12px 18px;text-align:center;flex:1;min-width:100px"><div style="color:{bc};font-family:Space Mono,monospace;font-size:1.5rem">{v}</div><div style="color:#6b8a92;font-size:0.7rem;margin-top:3px">{l}</div></div>'
          for v,l,bc,bg in [(clean,"CLEAN","#00e676","#00e67215"),(contam,"CONTAMINATED","#ffab00","#ffab0015"),
                            (nonrec,"NON-RECYCLABLE","#ff1744","#ff174415"),(len(results),"TOTAL","#18ffff","#18ffff10")]])}
        </div>""", unsafe_allow_html=True)

        for i in range(0, len(results), 2):
            cols = st.columns(2, gap="large")
            for j, col in enumerate(cols):
                if i+j < len(results):
                    r  = results[i+j]
                    sc = {"Clean Recyclable Waste":"#00e676","Contaminated / Mixed Waste":"#ffab00",
                          "Non-Recyclable Waste":"#ff1744"}.get(r["ana"]["status"],"#6b8a92")
                    with col:
                        st.markdown(f'<div style="color:#6b8a92;font-size:0.7rem;margin-bottom:4px">{r["name"]}</div>', unsafe_allow_html=True)
                        st.image(r["ann"], use_container_width=True)
                        st.markdown(f'<div style="background:{sc}15;border:1px solid {sc}45;border-radius:7px;padding:8px 12px;margin:5px 0;font-size:0.8rem;color:{sc}">{r["ana"]["status"]} | {r["ana"]["contamination_percent"]}% contamination</div>', unsafe_allow_html=True)

# ══ TAB 4 — ANALYTICS ════════════════════════════════════════
with tab4:
    st.markdown(section_header("Session Analytics","DASHBOARD"), unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown("""<div style="background:#111518;border:1px dashed #1f2a2e;border-radius:12px;
            padding:48px;text-align:center;margin-top:16px">
            <div style="font-size:2rem;margin-bottom:10px">📊</div>
            <div style="color:#6b8a92;font-size:0.88rem">No detections yet.<br>
            Use Live Webcam, Upload, or Batch tabs to begin.</div></div>""", unsafe_allow_html=True)
    else:
        df = pd.DataFrame(st.session_state.history)
        total = len(df); avg_c = df["contamination_%"].mean()
        k1,k2,k3,k4,k5,k6 = st.columns(6)
        k1.metric("Total Scans",    total)
        k2.metric("Avg Contam.",    f"{avg_c:.1f}%")
        k3.metric("Clean Scans",    int((df["status"]=="Clean Recyclable Waste").sum()))
        k4.metric("High Risk",      int((df["risk_level"]=="High").sum()))
        k5.metric("Recyclable",     int(df["recyclable"].sum()))
        k6.metric("Non-Recyclable", int(df["non_recyclable"].sum()))

        st.markdown("<br>", unsafe_allow_html=True)
        ca, cb = st.columns(2, gap="large")
        with ca:
            st.markdown(section_header("Status Breakdown","CHART"), unsafe_allow_html=True)
            st.bar_chart(df["status"].value_counts(), color="#00e676")
        with cb:
            st.markdown(section_header("Contamination Trend","OVER TIME"), unsafe_allow_html=True)
            st.line_chart(df.reset_index(drop=True)[["contamination_%"]], color="#ffab00")
        st.markdown(section_header("Risk Distribution","CHART"), unsafe_allow_html=True)
        st.bar_chart(df["risk_level"].value_counts(), color="#ff1744")
        st.markdown(section_header("Full Detection Log","TABLE"), unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ══ TAB 5 — AI TIPS ══════════════════════════════════════════
with tab5:
    st.markdown(section_header("AI Waste Reduction Tips","SMART GUIDANCE"), unsafe_allow_html=True)

    tips = {
        "plastic_bottle":          ("♻️ Plastic Bottle",    "Rinse before recycling. Remove caps — recycle separately. Crush to save bin space."),
        "can":                     ("🥫 Metal Can",          "Aluminium cans are infinitely recyclable. Rinse lightly — do not crush."),
        "cardboard_box":           ("📦 Cardboard Box",      "Flatten boxes. Remove tape and staples. Keep dry — wet cardboard is rejected."),
        "battery":                 ("🔋 Battery",            "NEVER put in regular trash. Take to a battery recycling drop-off — toxic chemicals leak into soil."),
        "plastic_bag":             ("🛍️ Plastic Bag",        "Most kerbside bins don't accept plastic bags. Use supermarket collection points."),
        "snack_bag":               ("🍫 Snack Bag",          "Multi-layer snack bags are not recyclable. Dispose in general waste."),
        "light_bulb":              ("💡 Light Bulb",         "CFL bulbs contain mercury — hazardous waste only. LEDs go to electronics recycling."),
        "chemical_plastic_gallon": ("🧴 Chemical Container", "Take to hazardous waste facility. Never pour chemicals down the drain."),
        "scrap_paper":             ("📄 Scrap Paper",        "Highly recyclable. Avoid waxed or laminated paper. Shred sensitive docs first."),
        "straw":                   ("🥤 Straw",              "Too small for most recycling sorters. Switch to reusable metal or bamboo straws."),
    }
    general = [
        ("🔁 Reduce First",           "The best waste is waste never created. Reduce before recycling."),
        ("🚿 Clean Before Recycling", "Food contamination is the #1 reason recyclables are rejected at plants."),
        ("🏷️ Check the Label",       "Look for recycling symbols (♻️ 1–7). Not all plastics are accepted everywhere."),
        ("📅 Segregate Daily",        "One contaminated bin can reject an entire batch at the recycling plant."),
        ("🌍 Local Rules Matter",     "Recycling rules vary by city — check your municipal guidelines."),
    ]

    if st.session_state.history:
        recent = []
        for h in st.session_state.history[-5:]:
            for item in h["items"].split(", "): recent.append(item.strip())
        matched = {k:v for k,v in tips.items() if k in recent}
        if matched:
            st.markdown(section_header("Tips for Recently Detected Items","PERSONALISED"), unsafe_allow_html=True)
            for title, tip in matched.values():
                st.markdown(f'<div style="background:#111518;border:1px solid #1f2a2e;border-left:3px solid #00e676;border-radius:0 10px 10px 0;padding:14px 16px;margin-bottom:9px"><div style="color:#00e676;font-family:Space Mono,monospace;font-size:0.8rem;font-weight:700;margin-bottom:5px">{title}</div><div style="color:#aaa;font-size:0.82rem;line-height:1.65">{tip}</div></div>', unsafe_allow_html=True)

    st.markdown(section_header("General Best Practices","GUIDE"), unsafe_allow_html=True)
    for title, tip in general:
        st.markdown(f'<div style="background:#111518;border:1px solid #1f2a2e;border-left:3px solid #18ffff;border-radius:0 10px 10px 0;padding:14px 16px;margin-bottom:9px"><div style="color:#18ffff;font-family:Space Mono,monospace;font-size:0.8rem;font-weight:700;margin-bottom:5px">{title}</div><div style="color:#aaa;font-size:0.82rem;line-height:1.65">{tip}</div></div>', unsafe_allow_html=True)

    st.markdown(section_header("Complete Class Reference","ALL 22 CLASSES"), unsafe_allow_html=True)
    rc, nc = st.columns(2, gap="large")
    with rc:
        st.markdown('<div style="color:#00e676;font-family:Space Mono,monospace;font-size:0.7rem;letter-spacing:.1em;margin-bottom:8px">♻️ RECYCLABLE</div>', unsafe_allow_html=True)
        for c in ["can","cardboard_bowl","cardboard_box","plastic_bag","plastic_bottle","plastic_bottle_cap","plastic_box","plastic_cutlery","plastic_cup","plastic_cup_lid","reusable_paper","scrap_paper","scrap_plastic"]:
            st.markdown(f'<div style="background:#00e67210;border:1px solid #00e67228;border-radius:6px;padding:7px 11px;margin-bottom:5px;font-size:0.8rem;color:#ccc">{c.replace("_"," ").title()}</div>', unsafe_allow_html=True)
    with nc:
        st.markdown('<div style="color:#ff1744;font-family:Space Mono,monospace;font-size:0.7rem;letter-spacing:.1em;margin-bottom:8px">🚫 NON-RECYCLABLE</div>', unsafe_allow_html=True)
        for c in ["battery","chemical_plastic_bottle","chemical_plastic_gallon","chemical_spray_can","light_bulb","paint_bucket","snack_bag","stick","straw"]:
            st.markdown(f'<div style="background:#ff174410;border:1px solid #ff174428;border-radius:6px;padding:7px 11px;margin-bottom:5px;font-size:0.8rem;color:#ccc">{c.replace("_"," ").title()}</div>', unsafe_allow_html=True)
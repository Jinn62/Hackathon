"""
app.py — 나의 터빈일지 | 풍력 터빈 손상 탐지 관리자 알람 대시보드

실행:
    pip install streamlit ultralytics
    streamlit run app.py

준비물: best.pt (학습된 모델), severity.py (같은 폴더)
"""
import base64
import shutil
import subprocess
import time
from io import BytesIO
from datetime import datetime

import numpy as np
import requests
import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from config import MODEL_PATH
from history_store import add_history_entry, find_entry, load_history
from severity import (
    assess_image, assess_video,
    CLASSES, GRADE_COLOR, GRADE_BG, GRADE_BORDER,
    GRADE_ACTION, GRADE_GUIDE, CLASS_ICON, CLASS_WEIGHT,
    size_label, score_one,
)

# ===== 설정 =====
DISPLAY_CONF_MIN = 0.7          # 화면에 표시할 최소 신뢰도 (판정 로직과는 무관, 표시 전용)
NTFY_TOPIC = "turbine-alarm-1234"
IMAGE_TYPES = ["jpg", "jpeg", "png"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]
VIDEO_SAMPLE_INTERVAL_SEC = 1.0  # 동영상에서 프레임을 샘플링할 간격
VIDEO_MAX_FRAMES = 30            # 동영상당 최대 분석 프레임 수 (데모 환경 처리 시간 보호)
VIDEO_OUTPUT_FPS = 2             # 탐지 결과 타임랩스 영상 재생 속도 (샘플링 간격과 무관하게 고정)


def encode_frames_to_video(frames_bgr, fps, output_path):
    """BGR numpy 프레임들을 ffmpeg(H.264)로 인코딩해 브라우저 재생 가능한 mp4로 저장한다.

    opencv-python-headless는 라이선스 문제로 H.264 인코딩을 지원하지 않아
    (기본 mp4v 코덱은 Chrome/Firefox에서 재생되지 않음) 시스템 ffmpeg를 직접 사용한다.
    """
    if not frames_bgr:
        raise ValueError("인코딩할 프레임이 없습니다.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg를 찾을 수 없습니다. 로컬은 ffmpeg를 설치하고, "
            "Streamlit Cloud는 packages.txt에 ffmpeg를 추가하세요."
        )

    height, width = frames_bgr[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    raw = b"".join(np.ascontiguousarray(f).tobytes() for f in frames_bgr)
    proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 인코딩 실패: {proc.stderr.decode(errors='ignore')[-500:]}")

st.set_page_config(page_title="나의 터빈일지", page_icon="🌀", layout="wide")


# ===== 스타일 (산업용 대시보드 톤 — 라이트 테마 고정) =====
st.markdown(
    """
    <style>
    /* ---------- 전역 배경/텍스트 (다크모드 자동전환으로 인한 흰 텍스트 방지) ---------- */
    .stApp { background: #F4F7FB; color: #111827; }
    section.main > div { padding-top: 1.2rem; padding-bottom: 2.5rem; }
    section.main .block-container {
        width: 100%; max-width: 1600px !important; margin: 0 auto;
        padding-left: 40px !important; padding-right: 40px !important;
    }
    .st-key-result_dashboard {
        width: 100%; margin: 0 auto;
    }
    .st-key-result_dashboard [data-testid="stHorizontalBlock"]:has(.detection-card-anchor):has(.action-panel-anchor) {
        width: 100%; display: grid !important;
        grid-template-columns: minmax(0, 1.8fr) minmax(380px, .9fr) !important;
        gap: 28px !important; align-items: stretch;
    }
    .st-key-result_dashboard [data-testid="stHorizontalBlock"]:has(.detection-card-anchor):has(.action-panel-anchor) > [data-testid="stColumn"] {
        width: 100% !important; min-width: 0 !important; height: 100%;
        display: flex; flex-direction: column; flex: none !important;
    }
    /* stColumn stretches to the grid row height natively, but its descendants
       (Streamlit's own VerticalBlock/LayoutWrapper wrappers) use flex-direction:
       column with flex-grow:0 by default, so they don't pass that height down.
       A flex-grow chain worked in Chrome but not Safari, so use single-cell CSS
       Grid instead — grid ROWS stretch to fill by default here, but an implicit
       grid COLUMN shrinks to content unless explicitly told to fill (each of
       these wrappers has exactly one child), so grid-template-columns:1fr is
       required too — width:100% alone isn't enough for the child's track. */
    .st-key-result_dashboard [data-testid="stHorizontalBlock"]:has(.detection-card-anchor):has(.action-panel-anchor) > [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
    .st-key-result_dashboard [data-testid="stHorizontalBlock"]:has(.detection-card-anchor):has(.action-panel-anchor) > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {
        display: grid; grid-template-columns: 1fr; min-height: 0; width: 100%;
    }
    .st-key-result_dashboard [data-testid="stHorizontalBlock"]:has(.detection-card-anchor):has(.action-panel-anchor) > [data-testid="stColumn"]:has(.action-panel-anchor) {
        min-width: 380px !important;
    }
    .st-key-detail_expander_wrap {
        width: 100%; margin: 18px auto 0;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp p, .stApp label, .stApp li { color: #111827; }
    [data-testid="stHeader"] { background: rgba(244, 247, 251, .92); }
    [data-testid="stToolbar"] { color: #334155; }

    .app-header { display: flex; align-items: center; gap: 10px; margin: 0 0 2px; }
    .app-title { color: #0F172A; font-size: 30px; font-weight: 800; margin: 0; line-height: 1.2; }
    .app-subtitle {
        color: #475569; font-size: 14px; margin-top: 2px; margin-bottom: 4px;
    }
    .app-subtitle .en { letter-spacing: .02em; }

    /* ---------- 섹션 제목 ---------- */
    .section-title {
        color: #111827; font-size: 17px; font-weight: 800; margin: 0 0 10px 0;
    }
    .section-sub { color: #64748B; font-size: 12.5px; margin-top: -6px; margin-bottom: 10px; }
    .input-card-title { color: #111827; font-size: 17px; font-weight: 800; margin-bottom: 2px; }
    .input-card-sub { color: #64748B; font-size: 12.5px; margin-bottom: 8px; }

    .fade-in { animation: fadeIn .45s ease-in; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ---------- 공통 카드 스타일 ---------- */
    .kpi-card, .severity-card, .esc-card, .img-card, .overview-card,
    .ai-card, .compare-card, .notification-card {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06); padding: 20px;
    }

    /* 관리자 Overview */
    .overview-wrap { margin: 18px 0; }
    .overview-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; }
    .overview-card { padding: 16px 18px; box-shadow: 0 2px 10px rgba(15,23,42,.05); }
    .overview-label { color: #64748B; font-size: 11px; font-weight: 700; letter-spacing: .04em; }
    .overview-value { color: #0F172A; font-size: 22px; font-weight: 800; margin-top: 5px; }
    .overview-value.grade { color: var(--grade-color); }

    /* AI 의견 / 비교 / 알림 */
    .ai-card { border-left: 5px solid #2563EB; margin: 0 0 16px; padding: 20px; background: #F8FAFC; box-shadow: 0 4px 16px rgba(15,23,42,.06); }
    .ai-title, .compare-title, .notification-title {
        color: #0F172A; font-size: 16px; font-weight: 700; margin-bottom: 8px;
    }
    .ai-copy { color: #334155; font-size: 14px; line-height: 1.7; }
    .notification-card { margin: 16px 0; padding: 14px 16px; background: #F8FAFC; border-color: #E2E8F0; }
    .notification-result { color: var(--grade-color); font-size: 13px; font-weight: 800; }
    .notification-meta { color: #64748B; font-size: 11px; margin-top: 4px; }
    .compare-card { width: 100%; max-width: none; margin: 12px 0 16px; padding: 24px; }
    .compare-table { display: grid; grid-template-columns: 1fr 1fr 1fr; width: 100%; border: 1px solid #E2E8F0; border-radius: 12px; overflow: hidden; }
    .compare-cell { padding: 10px 14px; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; color: #334155; font-size: 13px; }
    .compare-cell:nth-child(3n) { border-right: 0; }
    .compare-cell:nth-last-child(-n+3) { border-bottom: 0; }
    .compare-head { background: #F8FAFC; color: #64748B; font-size: 11px; font-weight: 800; }
    .compare-current { color: #0F172A; font-weight: 800; }
    .compare-delta { color: var(--delta-color); font-size: 14px; font-weight: 800; margin-top: 3px; }
    .compare-message { margin-top: 14px; padding: 12px 14px; border-radius: 10px; background: var(--grade-bg); color: #334155; font-size: 13px; line-height: 1.6; }
    .demo-tag { margin-left: 7px; border-radius: 999px; background: #EFF6FF; color: #2563EB; padding: 2px 7px; font-size: 10px; }
    .priority-card { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-top: 14px; padding: 20px 22px; background: linear-gradient(135deg,#0F172A,#1E3A5F); border: 1px solid #334155; border-radius: 16px; box-shadow: 0 6px 20px rgba(15,23,42,.16); }
    .priority-eyebrow { color: #94A3B8; font-size: 11px; font-weight: 700; letter-spacing: .06em; }
    .priority-id { color: #FFFFFF; font-size: 23px; font-weight: 800; margin-top: 5px; }
    .priority-metrics { display: flex; align-items: center; gap: 18px; }
    .priority-metric { color: #E2E8F0; font-size: 13px; }
    .priority-metric b { color: #FFFFFF; font-size: 17px; margin-left: 5px; }
    .priority-status { color: var(--grade-color); background: #FFFFFF; border-radius: 999px; padding: 7px 12px; font-size: 12px; font-weight: 800; }
    .brief-list { margin: 0; padding-left: 18px; color: #334155; font-size: 14px; line-height: 1.55; }
    .brief-list li { margin-bottom: 4px; }
    .brief-list li:last-child { margin-bottom: 0; }
    .brief-actions { margin-top: 12px; padding-top: 11px; border-top: 1px solid #DBEAFE; }
    .brief-actions-grid { display: grid; grid-template-columns: 1fr 1fr; column-gap: 10px; row-gap: 2px; }
    .brief-actions-title { color: #1E3A8A; font-size: 12px; font-weight: 800; margin-bottom: 7px; }
    .brief-action { color: #334155; font-size: 13px; line-height: 1.6; }
    .alert-checklist { display: grid; gap: 6px; }
    .alert-done { color: #166534; font-size: 13px; font-weight: 700; }
    .alert-off { color: #94A3B8; font-size: 13px; }
    .report-section { margin-top: 12px; padding-top: 12px; border-top: 1px solid #E2E8F0; }
    .report-section-title { color: #0F172A; font-size: 14px; font-weight: 800; }
    .report-section-sub { color: #64748B; font-size: 11px; margin-top: 2px; }
    .st-key-side_panel {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
        box-shadow: 0 4px 16px rgba(15,23,42,.06); padding: 20px;
        width: 100%; min-width: 380px; height: 100%;
        display: flex; flex-direction: column; gap: 16px;
    }
    .st-key-side_panel > [data-testid="stVerticalBlock"],
    .st-key-side_panel > div > [data-testid="stVerticalBlock"] { gap: 20px; }
    .st-key-side_panel .severity-card {
        width: 100%; padding: 20px; border: 1px solid #E2E8F0; border-radius: 16px;
        box-shadow: 0 4px 16px rgba(15,23,42,.06); margin-bottom: 0;
    }
    .st-key-side_panel .ai-card { margin-bottom: 0; }
    /* Streamlit gives stMarkdownContainer a built-in -16px margin-bottom that
       cancels the flex `gap` above — neutralize it so cards keep real spacing */
    .st-key-side_panel [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
    .st-key-detection_card [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
    .st-key-detection_card > [data-testid="stVerticalBlock"],
    .st-key-detection_card > div > [data-testid="stVerticalBlock"] { gap: 16px; }
    [data-testid="stExpander"]:has(.detail-anchor) {
        margin-top: 12px; margin-bottom: 12px;
        border: 1px solid #E2E8F0 !important; border-radius: 16px !important;
        box-shadow: 0 4px 16px rgba(15,23,42,.06); overflow: hidden;
    }
    [data-testid="stExpander"]:has(.detail-anchor) summary {
        padding: 16px 22px; background: #F8FAFC;
    }
    [data-testid="stExpander"]:has(.detail-anchor) summary:hover { background: #F1F5F9; }
    [data-testid="stExpander"]:has(.detail-anchor) summary p {
        font-size: 14.5px; font-weight: 800; color: #0F172A;
    }
    [data-testid="stExpander"]:has(.detail-anchor) [data-testid="stExpanderDetails"] {
        padding: 22px; border-top: 1px solid #E2E8F0; background: #FFFFFF;
    }
    .st-key-stats_wrap {
        padding: 16px 0 28px;
    }
    .st-key-stats_wrap .kpi-card {
        margin-top: 16px;
    }
    [data-testid="stDownloadButton"] button { min-height: 46px; font-weight: 800; border-radius: 10px; }
    .action-grid-anchor { height: 0; }
    .st-key-action_grid [data-testid="stHorizontalBlock"] {
        gap: 10px;
    }
    .st-key-action_grid button { min-height: 44px; height: 44px; }
    @media (max-width: 1100px) {
        .st-key-result_dashboard [data-testid="stHorizontalBlock"]:has(.detection-card-anchor):has(.action-panel-anchor) {
            grid-template-columns: minmax(0, 1fr) !important;
        }
        .st-key-result_dashboard [data-testid="stHorizontalBlock"]:has(.detection-card-anchor):has(.action-panel-anchor) > [data-testid="stColumn"]:has(.action-panel-anchor),
        .st-key-side_panel { min-width: 0 !important; }
        .st-key-detection_image_wrap [data-testid="stImage"] { height: auto; }
        .st-key-detection_image_wrap [data-testid="stImage"] img {
            height: auto !important; object-fit: contain;
        }
    }
    @media (max-width: 850px) {
        .overview-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .priority-card, .priority-metrics { align-items: flex-start; flex-direction: column; }
        .brief-actions-grid { grid-template-columns: 1fr; }
        section.main .block-container { padding-left: 18px !important; padding-right: 18px !important; }
        .st-key-result_dashboard [data-testid="stHorizontalBlock"]:has(.detection-card-anchor):has(.action-panel-anchor) {
            grid-template-columns: minmax(0, 1fr) !important;
        }
        .st-key-result_dashboard [data-testid="stHorizontalBlock"]:has(.detection-card-anchor):has(.action-panel-anchor) > [data-testid="stColumn"]:has(.action-panel-anchor) {
            min-width: 0 !important;
        }
    }

    /* KPI 카드 */
    .kpi-card {
        padding: 20px 18px;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(15,23,42,.10);
    }
    .kpi-icon  { font-size: 18px; }
    .kpi-label { font-size: 11px; color: #64748B; text-transform: uppercase;
                 letter-spacing: .06em; font-weight: 700; margin-top: 4px; }
    .kpi-value { font-size: 26px; font-weight: 800; color: #0F172A; margin-top: 2px; }

    /* 심각도 카드 (등급별 좌측 border 컬러는 --grade-color 로 주입) */
    .severity-card {
        background: #FFFFFF; border: 1px solid #E2E8F0;
        border-left: 6px solid var(--grade-color); box-shadow: 0 8px 24px rgba(15,23,42,.06);
    }
    .severity-card.grade-normal { border-left-color: #16A34A; background: #FFFFFF; }
    .severity-card.grade-watch { border-left-color: #D97706; background: #FFFFFF; }
    .severity-card.grade-caution { border-left-color: #EA580C; background: #FFFFFF; }
    .severity-card.grade-danger { border-left-color: #DC2626; background: #FFFFFF; }
    .severity-card.grade-normal .severity-grade { color: #16A34A; }
    .severity-card.grade-watch .severity-grade { color: #D97706; }
    .severity-card.grade-caution .severity-grade { color: #EA580C; }
    .severity-card.grade-danger .severity-grade { color: #DC2626; }
    .severity-top { display: flex; align-items: center; gap: 10px; }
    .severity-emoji { font-size: 26px; }
    .severity-grade { font-size: 22px; font-weight: 800; color: var(--grade-color); }
    .severity-score-row { margin-top: 10px; display: flex; align-items: baseline; gap: 8px; }
    .severity-score-label { font-size: 12px; color: #64748B; font-weight: 600; text-transform: uppercase; }
    .severity-score-value { font-size: 30px; font-weight: 800; color: #0F172A; }
    .severity-action {
        margin-top: 14px; font-size: 13px; color: #334155; background: #F8FAFC;
        border: none; border-radius: 10px; padding: 10px 12px;
    }

    /* 리포트(자동 보고서) 카드 — 어두운 카드이므로 흰 글씨 사용 (예외 허용 영역) */
    .report-card {
        background: #0F172A; color: #E2E8F0; border-radius: 16px; padding: 20px;
        margin-top: 18px; margin-bottom: 24px; font-size: 13px; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
    }
    .report-card .report-title {
        display: flex; justify-content: space-between; align-items: center;
        font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: #94A3B8;
        margin-bottom: 8px;
    }
    .report-card .report-badge {
        background: #1E293B; color: #38BDF8; padding: 2px 8px; border-radius: 999px; font-size: 10px;
    }
    .report-card .report-row { display: flex; justify-content: space-between; padding: 3px 0; }
    .report-card .report-row span:first-child { color: #94A3B8; }
    .report-card .report-row span:last-child { font-weight: 700; color: #F1F5F9; }

    /* 에스컬레이션 카드 */
    .esc-header { font-size: 14px; font-weight: 800; color: #0F172A; margin-bottom: 10px; }
    .esc-row { margin-bottom: 8px; }
    .esc-row-label { font-size: 11px; color: #64748B; font-weight: 700; text-transform: uppercase;
                      margin-right: 6px; }
    .chip {
        display: inline-flex; align-items: center; gap: 5px; background: #F1F5F9;
        border: 1px solid #E2E8F0; border-radius: 999px; padding: 3px 11px; margin: 3px 5px 0 0;
        font-size: 12.5px; font-weight: 600; color: #334155;
    }
    .esc-note {
        margin-top: 6px; font-size: 12.5px; font-weight: 600; border-radius: 8px;
        padding: 8px 12px; background: var(--grade-bg); color: var(--grade-color);
        border: 1px solid var(--grade-border);
    }

    /* 등급 범례 */
    .legend-row { display: flex; align-items: center; gap: 8px; font-size: 13px; padding: 3px 0; color: #111827; }
    .legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

    /* ---------- 점검 입력 카드 (max-width + 높이 정렬) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.input-card-anchor),
    div[data-testid="stLayoutWrapper"]:has(.input-card-anchor) {
        max-width: 960px; margin: 0 auto 8px;
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.input-card-anchor) > div,
    div[data-testid="stLayoutWrapper"]:has(.input-card-anchor) > div {
        background: #FFFFFF; border-radius: 16px;
    }
    /* 입력 컬럼: 68/28 비율, 동일한 기준선과 24px 간격 */
    [data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]):has([data-testid="stTextInput"]) {
        display: flex; align-items: flex-end; gap: 24px;
    }
    [data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]):has([data-testid="stTextInput"]) > div:first-child {
        flex: 0 1 68%; width: 68%;
    }
    [data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]):has([data-testid="stTextInput"]) > div:last-child {
        flex: 0 1 28%; width: 28%;
    }
    [data-testid="stFileUploader"], [data-testid="stTextInput"] {
        width: 100%;
    }
    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
    [data-testid="stTextInput"] [data-testid="stWidgetLabel"] {
        min-height: 24px; margin-bottom: 6px; display: flex; align-items: center;
    }
    [data-testid="stFileUploaderDropzone"] {
        box-sizing: border-box; min-height: 56px; height: 56px; padding: 0 16px;
        display: flex; align-items: center; background: #F8FAFC;
        border-color: #CBD5E1; border-radius: 12px;
    }
    [data-testid="stFileUploaderDropzone"] > div {
        width: 100%; display: flex; align-items: center; min-height: 0; padding: 0;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] svg { display: none; }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex; flex-direction: column; justify-content: center; min-height: 0;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small { font-size: 12px; }
    [data-testid="stFileUploader"] section { padding: 0; margin: 0; }
    [data-testid="stTextInput"] input {
        box-sizing: border-box; height: 56px; min-height: 56px; padding: 0 16px;
        line-height: 56px; font-size: 15px; background: #F8FAFC; color: #0F172A;
        border-color: #CBD5E1; border-radius: 12px !important;
    }
    [data-testid="stTextInput"] > div,
    [data-testid="stTextInput"] [data-baseweb="input"] {
        min-height: 56px; height: 56px; display: flex; align-items: center;
        border-radius: 12px;
    }
    [data-testid="stFileUploaderDropzone"] { margin-top: 0; }
    [data-testid="stWidgetLabel"] p { color: #334155; font-weight: 600; font-size: 13px; }
    div[data-testid="stFileUploaderFile"] {
        box-sizing: border-box; min-height: 56px; height: 56px; margin: 0; padding: 0 14px;
        display: flex; align-items: center; color: #111827; background: #F8FAFC;
        border: 1px solid #CBD5E1; border-radius: 12px;
    }
    div[data-testid="stFileUploaderFile"] > div {
        min-height: 0; display: flex; align-items: center;
    }
    div[data-testid="stFileUploaderFile"] button {
        align-self: center; margin: 0;
    }
    [data-testid="stFileUploader"]:has([data-testid="stFileUploaderFile"])
    [data-testid="stFileUploaderDropzone"] { display: none; }

    /* 심각도와 등급 안내를 시각적으로 분리 */
    .severity-card { margin-bottom: 14px; }
    [data-testid="stElementContainer"]:has(.severity-card) +
    [data-testid="stElementContainer"] [data-testid="stExpander"] { margin-top: 12px; }

    /* ---------- 탐지 결과 카드 ---------- */
    .st-key-detection_card {
        width: 100%; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
        box-shadow: 0 4px 16px rgba(15,23,42,.06); min-height: 0; height: 100%;
        display: flex; flex-direction: column; justify-content: center;
        padding: 28px !important;
    }
    .st-key-detection_card .section-title {
        margin: 0 0 18px; text-align: left; font-size: 18px; font-weight: 700;
    }
    .st-key-detection_image_wrap [data-testid="stImage"] {
        width: 100%; height: 520px; max-width: none; margin: 0 auto;
        display: flex; align-items: center; justify-content: center; overflow: hidden;
        background: #F8FAFC; border: 0; border-radius: 16px; padding: 0; box-shadow: none;
    }
    .st-key-detection_image_wrap [data-testid="stImageContainer"] {
        width: 100%; height: 100%;
    }
    .st-key-detection_image_wrap [data-testid="stImage"] img {
        display: block; max-height: none; max-width: 100%; width: 100% !important;
        height: 100% !important; margin: 0 auto; object-fit: cover; object-position: center;
        border-radius: 14px;
    }
    .st-key-detection_image_wrap [data-testid="stVideo"] {
        width: 100%; margin: 0 auto; border-radius: 16px; overflow: hidden;
        background: #0F172A;
    }
    .st-key-detection_image_wrap [data-testid="stVideo"] video {
        width: 100%; max-height: 520px; display: block; margin: 0 auto;
        object-fit: contain; border-radius: 16px;
    }
    .st-key-detection_image_wrap [data-testid="stCaptionContainer"],
    .st-key-detection_image_wrap .stCaption {
        width: 100%; max-width: none; margin: 12px auto 0; color: #64748B !important;
        font-size: 13px; text-align: center; justify-content: center;
    }
    .st-key-detection_image_wrap {
        width: 100%; max-width: 980px; margin: 0 auto;
    }
    @media (max-width: 1100px) {
        .st-key-detection_image_wrap [data-testid="stImage"] { height: auto; }
        .st-key-detection_image_wrap [data-testid="stImage"] img {
            height: auto !important; object-fit: contain;
        }
        .st-key-detection_card, .st-key-side_panel { height: auto; }
    }

    /* ---------- 데이터프레임(표) 라이트 스타일 ---------- */
    [data-testid="stDataFrame"] {
        border: 1px solid #E5E7EB; border-radius: 12px; overflow: hidden;
        background: #FFFFFF; color: #111827;
    }
    [data-testid="stDataFrame"] canvas { color-scheme: light; }
    [data-testid="stCheckbox"] p { color: #334155; }
    [data-testid="stExpander"] { background: #FFFFFF; border-color: #E5E7EB; }
    [data-testid="stExpander"] summary p { color: #334155; }
    hr { border-color: #E2E8F0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    # 탐지 이미지에 그려지는 라벨(model.plot())은 모델 내부 names를 그대로 쓴다.
    # 학습 시 클래스명 대소문자/표기가 달라도 대시보드 표기(CLASSES)와 항상 맞도록 덮어쓴다.
    # model.names는 읽기 전용 프로퍼티라 내부 model.model.names를 직접 바꿔야 한다.
    if set(model.names.keys()) == set(CLASSES.keys()):
        model.model.names = dict(CLASSES)
    return model


def channel_badges(channels):
    """알림 채널 문자열 → (아이콘, 라벨) 배지 목록 (표시 전용)"""
    badges = []
    for ch in channels:
        low = ch.lower()
        if "로그" in ch:
            badges.append(("📝", "로그 기록"))
        if "대시보드" in ch:
            badges.append(("📊", "Dashboard"))
        if "이메일" in ch:
            badges.append(("📧", "Email"))
        if "팀" in ch and "메시지" in ch:
            badges.append(("💬", "팀 메시지"))
        if "slack" in low:
            badges.append(("🚨", "Slack") if "즉시" in ch else ("💬", "Slack"))
        if "sms" in low:
            badges.append(("📱", "SMS"))
    return badges


def target_badges(targets):
    """알림 대상 문자열 → (아이콘, 라벨) 배지 목록 (표시 전용)"""
    badges = []
    for t in targets:
        if "현장" in t:
            badges.append(("👷", "현장 담당자"))
        elif t == "팀":
            badges.append(("👥", "팀"))
        elif "관리자" in t or "책임자" in t:
            badges.append(("👨‍💼", "관리자/책임자"))
        else:
            badges.append(("👤", t))
    return badges


def render_chips(badges):
    return "".join(f"<span class='chip'>{icon} {label}</span>" for icon, label in badges)


def kpi_card(icon, label, value):
    st.markdown(
        f"""<div class="kpi-card fade-in">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def notification_result(grade):
    """등급별 알림 시뮬레이션 문구."""
    return {
        "정상": "🟢 점검 로그 저장 완료",
        "관찰": "🟡 관리자 대시보드 등록 완료",
        "주의": "🟠 현장 담당자 이메일 발송 완료",
        "위험": "🔴 Slack 긴급알림 발송 완료",
    }[grade]


def make_ntfy_message(assessment):
    """ntfy 알림 본문 생성 — 등급에 따라 첫 줄만 바뀐다."""
    grade = assessment["grade"]

    level_map = {
        "정상": "🟢 정상 (Lv0)",
        "관찰": "🟡 관찰 (Lv1)",
        "주의": "⚠️ 주의 (Lv2)",
        "위험": "🔴 위험 (Lv3)",
    }

    action_map = {
        "정상": "정상 운영 유지",
        "관찰": "정기 점검 시 확인",
        "주의": "24시간 내 현장 점검 계획 수립",
        "위험": "즉시 현장 확인 및 관리자 보고",
    }

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""{level_map[grade]}

터빈 : {assessment["turbine_id"]}
위험도 : {assessment["score"]}

Damage : {assessment["n_damage"]}건
Dirt : {assessment["n_dirt"]}건
재확인 : {assessment["recheck"]}건

권장조치
{action_map[grade]}

{now}
"""


def send_ntfy_alert(assessment):
    """현재 분석 결과를 ntfy 푸시 알림으로 발송한다."""
    url = f"https://ntfy.sh/{NTFY_TOPIC}"

    title = f"{assessment['emoji']} {assessment['grade']} - {assessment['turbine_id']}"
    message = make_ntfy_message(assessment)

    headers = {
        # HTTP header values must be latin-1; Title has Korean text and an emoji,
        # so it must go over the wire as UTF-8 bytes or requests raises
        # UnicodeEncodeError when sending (ntfy decodes byte header values as UTF-8).
        "Title": title.encode("utf-8"),
        "Tags": "warning,wind_turbine",
        "Priority": "4" if assessment["grade"] in ["주의", "위험"] else "3",
    }

    response = requests.post(
        url,
        data=message.encode("utf-8"),
        headers=headers,
        timeout=5,
    )
    response.raise_for_status()
    return True


def ai_briefing(result, previous):
    """관리자가 빠르게 읽을 수 있는 점검 요약과 우선 작업을 생성한다."""
    score_delta = result["score"] - previous["위험도"]
    if score_delta > 0:
        trend = "이전 점검 대비 위험도가 증가했습니다."
    elif score_delta < 0:
        trend = "이전 점검 대비 위험도가 감소했습니다."
    else:
        trend = "이전 점검 대비 위험도가 유지되었습니다."
    summary = [
        f"Damage {result['n_damage']}건이 탐지되었습니다.",
        f"Dirt {result['n_dirt']}건이 탐지되었습니다." if result["n_dirt"] else "Dirt는 탐지되지 않았습니다.",
        trend,
        "재확인 대상이 존재합니다." if result["recheck"] else "재확인 대상은 없습니다.",
    ]
    actions = [
        "탐지 영역 현장 확인",
        "점검 계획에 탐지 결과 반영",
        "재확인 대상 우선 검토",
        "다음 점검 결과와 변화 비교",
    ]
    return summary, actions


def comparison_message(result, previous):
    score_delta = result["score"] - previous["위험도"]
    damage_delta = result["n_damage"] - previous["Damage"]
    if damage_delta > 0:
        return "⚠ 손상이 증가하고 있습니다. 점검 우선순위를 높이는 것을 권장합니다."
    if score_delta > 0:
        return "⚠ 위험도가 상승했습니다. 동일 부위의 변화 추적과 현장 확인을 권장합니다."
    if score_delta < 0:
        return "✓ 위험도가 감소했습니다. 정기 점검을 통해 개선 추세를 확인하세요."
    return "✓ 이전 점검과 동일한 수준입니다. 현재 점검 주기를 유지하세요."


def make_inspection_pdf(result, plotted_image, inspected_at):
    """현장 제출용 PDF 점검보고서를 메모리에서 생성한다."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        Image as ReportImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font_name = "HYSMyeongJo-Medium"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))

    pdf_buffer = BytesIO()
    document = SimpleDocTemplate(
        pdf_buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "KoreanTitle", parent=styles["Title"], fontName=font_name,
        fontSize=20, leading=27, textColor=colors.HexColor("#0F172A"),
        alignment=TA_CENTER, spaceAfter=4 * mm,
    )
    subtitle_style = ParagraphStyle(
        "KoreanSubtitle", parent=styles["Normal"], fontName=font_name,
        fontSize=11, leading=16, textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER, spaceAfter=8 * mm,
    )
    body_style = ParagraphStyle(
        "KoreanBody", parent=styles["Normal"], fontName=font_name,
        fontSize=10, leading=15, textColor=colors.HexColor("#111827"),
    )

    story = [
        Paragraph("나의 터빈일지", title_style),
        Paragraph("풍력터빈 점검보고서 · 현장 제출용", subtitle_style),
    ]
    report_data = [
        ["점검일시", inspected_at.strftime("%Y-%m-%d %H:%M:%S")],
        ["터빈 ID", result["turbine_id"]],
        ["Damage", f"{result['n_damage']}건"],
        ["Dirt", f"{result['n_dirt']}건"],
        ["위험도", f"{result['score']:.1f}"],
        ["등급", result["grade"]],
        ["권장조치", GRADE_ACTION[result["grade"]]],
    ]
    table = Table(report_data, colWidths=[38 * mm, 118 * mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F8FAFC")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#334155")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#111827")),
        ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([table, Spacer(1, 8 * mm), Paragraph("탐지 결과 이미지", body_style), Spacer(1, 3 * mm)])

    image_buffer = BytesIO()
    Image.fromarray(plotted_image).save(image_buffer, format="PNG")
    image_buffer.seek(0)
    report_image = ReportImage(image_buffer)
    max_width, max_height = 156 * mm, 105 * mm
    scale = min(max_width / report_image.imageWidth, max_height / report_image.imageHeight)
    report_image.drawWidth = report_image.imageWidth * scale
    report_image.drawHeight = report_image.imageHeight * scale
    story.append(report_image)
    document.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def image_to_base64_jpeg(image_rgb, quality=85):
    """RGB numpy 배열 → base64 JPEG 문자열 (공유 이력에 대표 이미지로 저장)."""
    buffer = BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def base64_jpeg_to_image(b64_str):
    """base64 JPEG 문자열 → RGB numpy 배열 (공유 이력에서 불러올 때 복원)."""
    return np.array(Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB"))


# ===== 헤더 =====
st.markdown(
    "<div class='app-header'><span class='app-title'>🌀 나의 터빈일지</span></div>"
    "<div class='app-subtitle'>AI 기반 풍력터빈 손상 탐지 · "
    "<span class='en'>Predictive Maintenance Dashboard</span></div>",
    unsafe_allow_html=True,
)
st.divider()

model = load_model()

# ===== 입력 =====
with st.container(border=True):
    st.markdown(
        "<div class='input-card-anchor'></div>"
        "<div class='input-card-title'>점검 입력</div>"
        "<div class='input-card-sub'>점검 이미지·영상과 관리 대상 터빈을 입력하세요.</div>",
        unsafe_allow_html=True,
    )
    col_in, col_id = st.columns([7, 3], gap="medium")
    with col_in:
        uploaded = st.file_uploader(
            "터빈 이미지/영상 업로드", type=IMAGE_TYPES + VIDEO_TYPES,
            help="영상은 1초 간격으로 최대 30프레임을 샘플링해 가장 위험도가 높은 "
                 "프레임을 대표 결과로 분석합니다.",
        )
    with col_id:
        turbine_id = st.text_input("터빈 ID", value="터빈-03")

def render_analysis(result, turbine_id, plotted, inspected_at, previous, is_demo_previous, is_history_view=False):
    """분석 결과 하나를 대시보드로 렌더링한다.

    실시간 업로드 직후와, 공유 이력에서 과거 기록을 불러와 다시 볼 때
    양쪽에서 재사용한다 (관리자 A가 분석한 화면을 B/C가 그대로 다시 볼 수
    있어야 하므로, 렌더링 로직 자체를 입력값에서 완전히 분리했다).
    """
    grade = result["grade"]
    color = GRADE_COLOR[grade]
    grade_bg, grade_border = GRADE_BG[grade], GRADE_BORDER[grade]
    css_vars = f"--grade-color:{color};--grade-bg:{grade_bg};--grade-border:{grade_border};"
    grade_class = {
        "정상": "grade-normal", "관찰": "grade-watch",
        "주의": "grade-caution", "위험": "grade-danger",
    }[grade]

    if is_history_view:
        st.markdown(
            f"""<div class="notification-card fade-in" style="border-left:4px solid #2563EB;">
                    🔍 <b>불러온 점검 기록</b> — {turbine_id} · {inspected_at.strftime('%Y-%m-%d %H:%M')}
                    (다른 관리자가 분석한 결과일 수 있습니다)
                </div>""",
            unsafe_allow_html=True,
        )
        st.write("")

    # ===== 관리자 Overview =====
    st.markdown(
        f"""<div class="overview-wrap fade-in" style="{css_vars}">
                <div class="section-title">관리자 Overview</div>
                <div class="overview-grid">
                    <div class="overview-card"><div class="overview-label">오늘 점검</div><div class="overview-value">1기</div></div>
                    <div class="overview-card"><div class="overview-label">현재 위험</div><div class="overview-value grade">{result['emoji']} {grade}</div></div>
                    <div class="overview-card"><div class="overview-label">Damage</div><div class="overview-value">{result['n_damage']}건</div></div>
                    <div class="overview-card"><div class="overview-label">Dirt</div><div class="overview-value">{result['n_dirt']}건</div></div>
                    <div class="overview-card"><div class="overview-label">재확인</div><div class="overview-value">{result['recheck']}건</div></div>
                </div>
                <div class="priority-card">
                    <div><div class="priority-eyebrow">★★★★★ 오늘 우선 점검 대상</div><div class="priority-id">{turbine_id}</div></div>
                    <div class="priority-metrics">
                        <div class="priority-metric">위험도 <b>{result['score']:.1f}</b></div>
                        <div class="priority-status">{result['emoji']} {grade}</div>
                    </div>
                </div>
            </div>""",
        unsafe_allow_html=True,
    )

    # ===== 결과: 좌(이미지) 우(판정) =====
    result_dashboard = st.container(key="result_dashboard")
    col_img, col_res = result_dashboard.columns([2, 1], gap="medium")
    detail_expander_host = st.container(key="detail_expander_wrap")

    with col_img:
        with st.container(border=True, key="detection_card"):
            st.markdown(
                "<div class='detection-card-anchor'></div>"
                "<div class='section-title'>탐지 결과</div>",
                unsafe_allow_html=True,
            )
            with st.container(key="detection_image_wrap"):
                caption = f"※ 화면에는 신뢰도 {int(DISPLAY_CONF_MIN*100)}% 이상 탐지만 표시됩니다."
                if result.get("source") == "video":
                    if result.get("output_video_path") and not is_history_view:
                        st.video(result["output_video_path"])
                        caption += (
                            f" (샘플링 {result['sampled_frames']}프레임 타임랩스 · "
                            f"위험도 최고 지점: {result['timestamp_sec']}초)"
                        )
                    elif is_history_view:
                        st.image(plotted, width="stretch")
                        caption += (
                            f" (영상 {result['timestamp_sec']}초 지점 · 위험도 최고 프레임 — "
                            "공유 이력에는 대표 프레임만 저장됩니다)"
                        )
                    else:
                        st.image(plotted, width="stretch")
                        caption += (
                            f" (영상 {result['timestamp_sec']}초 지점 · "
                            f"샘플링 {result['sampled_frames']}프레임 중 위험도 최고 프레임 — "
                            "타임랩스 영상 생성 실패로 대표 프레임만 표시)"
                        )
                else:
                    st.image(plotted, width="stretch")
                st.caption(caption)

    with col_res:
        briefing_points, briefing_actions = ai_briefing(result, previous)
        pdf_bytes = make_inspection_pdf(result, plotted, inspected_at)
        safe_turbine_id = "".join(char for char in turbine_id if char.isalnum()) or "Turbine"
        with st.container(border=True, key="side_panel"):
            st.markdown(
                f"""<div class="action-panel-anchor"></div>
                    <div class="severity-card {grade_class} fade-in" style="{css_vars}">
                        <div class="severity-top"><span class="severity-emoji">{result['emoji']}</span><span class="severity-grade">{grade}</span></div>
                        <div class="severity-score-row"><span class="severity-score-label">위험도</span><span class="severity-score-value">{result['score']}</span></div>
                        <div class="severity-action">권장 조치: {GRADE_ACTION[grade]}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

            summary_html = "".join(f"<li>{item}</li>" for item in briefing_points)
            actions_html = "".join(
                f"<div class='brief-action'>{index} {item}</div>"
                for index, item in zip(["①", "②", "③", "④"], briefing_actions)
            )
            st.markdown(
                f"""<div class="ai-card fade-in">
                        <div class="ai-title">🤖 AI 점검 브리핑</div>
                        <ul class="brief-list">{summary_html}</ul>
                        <div class="brief-actions"><div class="brief-actions-title">우선 권장 작업</div><div class="brief-actions-grid">{actions_html}</div></div>
                    </div>""",
                unsafe_allow_html=True,
            )

            dashboard_done = grade in ["관찰", "주의", "위험"]
            email_done = grade in ["주의", "위험"]
            sms_done = grade == "위험"
            st.markdown(
                "<div class='report-section'><div class='report-section-title'>다음 작업</div></div>",
                unsafe_allow_html=True,
            )
            with st.container(key="action_grid"):
                st.markdown("<div class='action-grid-anchor'></div>", unsafe_allow_html=True)
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    st.download_button(
                        "⬇ PDF 다운로드", data=pdf_bytes,
                        file_name=f"Inspection_Report_{safe_turbine_id}.pdf",
                        mime="application/pdf", width="stretch",
                    )
                with action_col2:
                    if st.button("🔔 담당자 알림", width="stretch", key="notify_manager"):
                        try:
                            send_ntfy_alert(result)
                            st.session_state.ntfy_status = ("success", None, False)
                        except Exception as e:
                            st.session_state.ntfy_status = ("error", str(e), False)

                ntfy_status = st.session_state.get("ntfy_status")
                if ntfy_status is None:
                    st.caption("외부 채널 연동 전")
                elif ntfy_status[0] == "success":
                    auto_suffix = " (위험 등급 자동 발송)" if ntfy_status[2] else ""
                    st.success(f"ntfy 담당자 알림 발송 완료{auto_suffix}")
                else:
                    auto_suffix = " (위험 등급 자동 발송 시도)" if ntfy_status[2] else ""
                    st.error(f"알림 발송 실패{auto_suffix}: {ntfy_status[1]}")

                if st.button("📅 현장점검 예약", width="stretch", key="schedule_inspection"):
                    st.toast("현장점검 예약 요청이 등록되었습니다. (시뮬레이션)")

        # 상세 정보는 필요할 때만 펼친다.
        with detail_expander_host.expander("📋 상세보기 · 등급 기준 / 계산 근거 / 검사 결과 / 통계"):
            st.markdown("<div class='detail-anchor'></div>", unsafe_allow_html=True)
            st.markdown(
                f"""<div class="notification-card" style="{css_vars}">
                        <div class="notification-title">알림 발송 결과</div>
                        <div class="alert-checklist">
                            <div class="{'alert-done' if dashboard_done else 'alert-off'}">{'✔' if dashboard_done else '○'} Dashboard {'등록 완료' if dashboard_done else '미등록'}</div>
                            <div class="{'alert-done' if email_done else 'alert-off'}">{'✔' if email_done else '○'} Email {'발송 완료' if email_done else '미발송'}</div>
                            <div class="{'alert-done' if sms_done else 'alert-off'}">{'✔' if sms_done else '○'} SMS {'발송 완료' if sms_done else '미사용'}</div>
                        </div>
                    </div>""",
                unsafe_allow_html=True,
            )
            st.markdown("**등급 기준**")
            for g in ["정상", "관찰", "주의", "위험"]:
                st.markdown(
                    f"<div class='legend-row'><span class='legend-dot' style='background:{GRADE_COLOR[g]}'></span><b>{g}</b> — {GRADE_GUIDE[g]}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("**위험도 계산 근거**")
            st.caption("객체별 점수 = 클래스 가중치 × (1 + Bounding Box 면적비 × 5) × 모델 신뢰도")
            score_rows = []
            for index, detection in enumerate(result["counted"], start=1):
                score_rows.append({
                    "객체": f"{detection['label']} #{index}", "가중치": CLASS_WEIGHT[detection["cls"]],
                    "면적비": f"{detection['area_ratio'] * 100:.2f}%", "신뢰도": f"{detection['conf'] * 100:.1f}%",
                    "기여 점수": round(score_one(detection["cls"], detection["area_ratio"], detection["conf"]), 1),
                })
            if score_rows:
                st.dataframe(pd.DataFrame(score_rows), width="stretch", hide_index=True, row_height=32)
            else:
                st.caption("위험도에 반영된 고신뢰도 탐지가 없습니다.")
            st.markdown(f"**총 위험도: {result['score']:.1f}점**")

            st.markdown(
                f"""<div class="report-card">
                        <div class="report-title">검사 결과 <span class="report-badge">AUTO REPORT · Generated by AI</span></div>
                        <div class="report-row"><span>터빈</span><span>{turbine_id}</span></div>
                        <div class="report-row"><span>Damage / Dirt</span><span>{result['n_damage']} / {result['n_dirt']}건</span></div>
                        <div class="report-row"><span>위험도 / 등급</span><span>{result['score']} / {grade}</span></div>
                        <div class="report-row"><span>권장조치</span><span>{GRADE_GUIDE[grade]}</span></div>
                    </div>""",
                unsafe_allow_html=True,
            )
            with st.container(key="stats_wrap"):
                st.markdown("<div class='stats-anchor'></div>", unsafe_allow_html=True)
                stat1, stat2, stat3 = st.columns(3)
                with stat1:
                    kpi_card(CLASS_ICON["Damage"], "Damage", f"{result['n_damage']}건")
                with stat2:
                    kpi_card(CLASS_ICON["Dirt"], "Dirt", f"{result['n_dirt']}건")
                with stat3:
                    kpi_card("👁", "재확인", f"{result['recheck']}건")

    # ===== 이전 점검 대비 =====
    score_delta = result["score"] - previous["위험도"]
    delta_symbol = "▲" if score_delta > 0 else "▼" if score_delta < 0 else "―"
    delta_color = "#DC2626" if score_delta > 0 else "#16A34A" if score_delta < 0 else "#64748B"
    previous_tag = "<span class='demo-tag'>DEMO BASELINE</span>" if is_demo_previous else ""
    st.markdown(
        f"""<div class="compare-card fade-in" style="{css_vars}--delta-color:{delta_color};">
                <div class="compare-title">이전 점검 대비 {previous_tag}</div>
                <div class="compare-table">
                    <div class="compare-cell compare-head">항목</div><div class="compare-cell compare-head">이전</div><div class="compare-cell compare-head">현재</div>
                    <div class="compare-cell">위험도</div><div class="compare-cell">{previous['위험도']:.1f}</div><div class="compare-cell compare-current">{result['score']:.1f} <span class="compare-delta">{delta_symbol} {score_delta:+.1f}</span></div>
                    <div class="compare-cell">Damage</div><div class="compare-cell">{previous['Damage']}건</div><div class="compare-cell compare-current">{result['n_damage']}건</div>
                    <div class="compare-cell">Dirt</div><div class="compare-cell">{previous['Dirt']}건</div><div class="compare-cell compare-current">{result['n_dirt']}건</div>
                    <div class="compare-cell">판정</div><div class="compare-cell">{previous['등급']}</div><div class="compare-cell compare-current">{result['emoji']} {grade}</div>
                </div>
                <div class="compare-message">{comparison_message(result, previous)}</div>
            </div>""",
        unsafe_allow_html=True,
    )

    # ===== 탐지 상세 표 =====
    if result["counted"] or result["recheck_list"]:
        st.markdown("<div class='section-title'>탐지 상세</div>", unsafe_allow_html=True)
        rows = []
        for d in sorted(result["counted"], key=lambda d: -d["conf"]):
            rows.append({
                "클래스": f"{CLASS_ICON.get(d['label'], '')} {d['label']}",
                "크기": size_label(d["area_ratio"]),
                "신뢰도": f"{d['conf']*100:.0f}%",
                "처리": "✅ 반영",
            })
        for d in sorted(result["recheck_list"], key=lambda d: -d["conf"]):
            rows.append({
                "클래스": f"{CLASS_ICON.get(d['label'], '')} {d['label']}",
                "크기": size_label(d["area_ratio"]),
                "신뢰도": f"{d['conf']*100:.0f}%",
                "처리": "⚠ 재확인",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, row_height=32)


if uploaded:
    file_ext = uploaded.name.rsplit(".", 1)[-1].lower()
    is_video = file_ext in VIDEO_TYPES
    st.session_state.pop("selected_history_id", None)  # 새 업로드가 불러오기 화면보다 우선한다

    # ===== 순차 진행 상태 표시 =====
    with st.status("분석 진행 중...", expanded=True) as status:
        if is_video:
            st.write("🎞️ 영상 로드 중...")
            tmp_path = f"_tmp_upload.{file_ext}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            time.sleep(0.3)

            st.write(f"🔎 프레임 샘플링(최대 {VIDEO_MAX_FRAMES}개) 및 YOLO 탐지 진행 중...")
            result = assess_video(
                model, tmp_path, turbine_id=turbine_id,
                sample_interval_sec=VIDEO_SAMPLE_INTERVAL_SEC, max_frames=VIDEO_MAX_FRAMES,
            )
            yolo_res = result["yolo_result"]
            st.write(
                f"🖼️ 대표 프레임 선정 완료 — {result['timestamp_sec']}초 지점 "
                f"(총 {result['sampled_frames']}프레임 중 위험도 최고)"
            )
            time.sleep(0.3)

            st.write("🎬 탐지 결과 타임랩스 영상 생성 중...")
            frame_plots_bgr = [
                fr["yolo_result"][fr["yolo_result"].boxes.conf >= DISPLAY_CONF_MIN]
                .plot(line_width=2, font_size=13)
                for fr in result["frame_results"]
            ]
            output_video_path = "_tmp_detection_timelapse.mp4"
            try:
                encode_frames_to_video(frame_plots_bgr, VIDEO_OUTPUT_FPS, output_video_path)
                result["output_video_path"] = output_video_path
            except Exception as e:
                result["video_encode_error"] = str(e)
            time.sleep(0.3)
        else:
            st.write("🖼️ 이미지 로드 중...")
            img = Image.open(uploaded).convert("RGB")
            tmp_path = "_tmp_upload.jpg"
            img.save(tmp_path)
            time.sleep(0.3)

            st.write("🔎 YOLO 탐지 진행 중...")
            result = assess_image(model, tmp_path, turbine_id=turbine_id)
            yolo_res = result["yolo_result"]
            time.sleep(0.3)

        st.write("📐 심각도 계산 중...")
        time.sleep(0.3)

        grade = result["grade"]
        st.write(f"{result['emoji']} 등급 판정 완료 — {grade}")
        time.sleep(0.3)

        st.write("📢 에스컬레이션 정책 확인 중...")
        time.sleep(0.3)

        status.update(label="분석 완료", state="complete", expanded=False)

    st.write("")

    # 위험 등급은 담당자 알림 버튼을 누르지 않아도 ntfy 푸시를 자동 발송한다.
    # 같은 업로드+터빈ID 조합에서는 재실행(다른 버튼 클릭 등)마다 중복 발송되지
    # 않도록 file_id로 한 번만 보낸다.
    auto_alert_key = f"{turbine_id}:{uploaded.file_id}"
    if grade == "위험" and st.session_state.get("last_auto_alert_key") != auto_alert_key:
        try:
            send_ntfy_alert(result)
            st.session_state.ntfy_status = ("success", None, True)
        except Exception as e:
            st.session_state.ntfy_status = ("error", str(e), True)
        st.session_state.last_auto_alert_key = auto_alert_key

    inspected_at = datetime.now()
    shared_history = load_history()
    previous = next(
        (item for item in shared_history if item["터빈"] == turbine_id),
        {"위험도": 24.0, "Damage": 1, "Dirt": 2, "등급": "🟠 주의"},
    )
    is_demo_previous = not any(item["터빈"] == turbine_id for item in shared_history)

    # 화면 및 PDF에 동일한 고신뢰도 탐지 이미지를 사용
    display_res = yolo_res[yolo_res.boxes.conf >= DISPLAY_CONF_MIN]
    # 표시 전용 라벨 크기와 선 두께를 고정해 박스 가장자리의 가독성을 확보한다.
    plotted = display_res.plot(line_width=2, font_size=13)[:, :, ::-1]

    render_analysis(result, turbine_id, plotted, inspected_at, previous, is_demo_previous)

    # ===== 공유 점검 이력에 기록 (모든 관리자가 함께 봄) =====
    add_history_entry({
        "시각": inspected_at.strftime("%H:%M:%S"),
        "날짜": inspected_at.strftime("%Y-%m-%d"),
        "터빈": turbine_id,
        "등급": f"{result['emoji']} {grade}",
        "위험도": result["score"],
        "Damage": result["n_damage"],
        "Dirt": result["n_dirt"],
        "grade": result["grade"], "emoji": result["emoji"], "score": result["score"],
        "n_damage": result["n_damage"], "n_dirt": result["n_dirt"], "recheck": result["recheck"],
        "channels": result["channels"], "targets": result["targets"],
        "counted": result["counted"], "recheck_list": result["recheck_list"],
        "source": result.get("source", "image"),
        "timestamp_sec": result.get("timestamp_sec"),
        "sampled_frames": result.get("sampled_frames"),
        "inspected_at": inspected_at.isoformat(),
        "previous_snapshot": previous,
        "is_demo_previous": is_demo_previous,
        "plotted_image_b64": image_to_base64_jpeg(plotted),
    })

elif st.session_state.get("selected_history_id"):
    shared_history = load_history()
    entry = find_entry(shared_history, st.session_state["selected_history_id"])
    if entry is None:
        st.info("선택한 점검 기록을 더 이상 찾을 수 없습니다 (오래되어 정리되었을 수 있습니다).")
    else:
        result = {
            "turbine_id": entry["터빈"], "grade": entry["grade"], "emoji": entry["emoji"],
            "score": entry["score"], "n_damage": entry["n_damage"], "n_dirt": entry["n_dirt"],
            "recheck": entry["recheck"], "channels": entry["channels"], "targets": entry["targets"],
            "counted": entry["counted"], "recheck_list": entry["recheck_list"],
            "source": entry.get("source", "image"),
            "timestamp_sec": entry.get("timestamp_sec"),
            "sampled_frames": entry.get("sampled_frames"),
        }
        plotted = base64_jpeg_to_image(entry["plotted_image_b64"])
        inspected_at = datetime.fromisoformat(entry["inspected_at"])
        if st.button("✕ 불러오기 닫기"):
            st.session_state.pop("selected_history_id", None)
            st.rerun()
        render_analysis(
            result, entry["터빈"], plotted, inspected_at,
            entry["previous_snapshot"], entry.get("is_demo_previous", False),
            is_history_view=True,
        )

# ===== 점검 이력 (터빈일지 · 전체 관리자 공유) =====
st.divider()
st.markdown("<div class='section-title'>📒 점검 이력 (터빈일지 · 전체 관리자 공유)</div>", unsafe_allow_html=True)
shared_history = load_history()
if shared_history:
    st.caption(
        "이 목록은 서버에 저장되어 모든 관리자가 함께 봅니다. 행을 선택하고 "
        "아래 버튼을 누르면 그 점검 당시 화면을 그대로 불러와 볼 수 있습니다. "
        "(업로드된 파일이 있으면 불러오기보다 우선 표시되니, 업로드 목록에서 파일을 제거하세요.)"
    )
    df = pd.DataFrame(shared_history)
    if st.checkbox("위험도 높은 순으로 정렬"):
        df = df.sort_values("위험도", ascending=False)
    display_cols = [c for c in ["시각", "날짜", "터빈", "등급", "위험도", "Damage", "Dirt"] if c in df.columns]

    def _tint_row(row):
        for g, bg in GRADE_BG.items():
            if g in row["등급"]:
                return [f"background-color:{bg}"] * len(row)
        return [""] * len(row)

    event = st.dataframe(
        df[display_cols].style.apply(_tint_row, axis=1),
        width="stretch", hide_index=True, row_height=32,
        column_config={"위험도": st.column_config.NumberColumn("위험도", format="%.1f")},
        on_select="rerun", selection_mode="single-row", key="history_table",
    )

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows and not uploaded:
        selected_entry = df.iloc[selected_rows[0]]
        if st.button(f"🔍 선택한 점검 불러오기 ({selected_entry['터빈']} · {selected_entry['시각']})"):
            st.session_state["selected_history_id"] = selected_entry["id"]
            st.rerun()
else:
    st.caption("아직 점검 이력이 없습니다. 이미지나 영상을 업로드하세요.")

"""history_store.py — 여러 관리자가 함께 보는 점검 이력을 JSON 파일에 저장/조회한다.

st.session_state는 브라우저 세션(탭)마다 완전히 분리되어 있어서, 관리자 A가
분석한 결과를 관리자 B/C는 볼 수 없다. 이 모듈은 서버 디스크의 JSON 파일에
기록해서 같은 배포에 접속한 모든 관리자가 서로의 점검 이력을 조회하고
다시 불러와 볼 수 있게 한다.

주의: Streamlit Cloud 같은 컨테이너 배포는 재배포/리부트 시 파일시스템이
초기화되므로, 그 시점에는 이 파일도 함께 사라진다 (진짜 영구 저장이
필요하면 외부 데이터베이스가 필요하다).
"""
import json
import threading
import uuid
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parent / "history_store.json"
MAX_HISTORY = 200  # 파일이 무한정 커지지 않도록 최근 N건만 유지

_lock = threading.Lock()


def _read_unlocked():
    if not HISTORY_PATH.is_file():
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def load_history():
    """저장된 점검 이력을 최신순(등록 역순)으로 반환한다."""
    with _lock:
        return _read_unlocked()


def add_history_entry(entry):
    """새 점검 기록을 맨 앞에 추가한다. id가 없으면 자동으로 부여한다."""
    entry = dict(entry)
    entry.setdefault("id", str(uuid.uuid4()))
    with _lock:
        history = _read_unlocked()
        history.insert(0, entry)
        history = history[:MAX_HISTORY]
        tmp_path = HISTORY_PATH.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
        tmp_path.replace(HISTORY_PATH)  # 원자적 교체 — 동시 읽기 중 깨진 파일 방지
    return entry


def find_entry(history, entry_id):
    return next((item for item in history if item.get("id") == entry_id), None)

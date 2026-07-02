"""
severity.py — 풍력 터빈 손상 심각도 판정 + 에스컬레이션
나의 터빈일지 / 서비스 데모 핵심 모듈

사용법:
    from ultralytics import YOLO
    from severity import assess_image

    model = YOLO("best.pt")
    result = assess_image(model, "test.jpg", turbine_id="터빈-03")
    print(result["message"])
"""

# ===== 설정 (나중에 테스트 후 조정) =====
CLASSES = {0: "Dirt", 1: "Damage"}
CLASS_WEIGHT = {0: 2, 1: 10}      # 손상이 오염보다 5배 위험

CONF_HIGH = 0.7                    # 이상: 확실한 탐지
CONF_LOW  = 0.4                    # 이하: 오탐 가능 → 무시

# 등급 임계치 (점수 → 등급)
#   0 정상 / 0<score<15 관찰 / 15<=score<40 주의 / 40<= 위험

# 에스컬레이션 정책 (등급 → 채널·대상)
ESCALATION = {
    "정상": {"channels": ["로그 기록"],                        "targets": []},
    "관찰": {"channels": ["대시보드"],                          "targets": ["현장 담당자"]},
    "주의": {"channels": ["대시보드", "이메일/팀 메시지"],       "targets": ["현장 담당자", "팀"]},
    "위험": {"channels": ["대시보드", "즉시 Slack/SMS 경보"],    "targets": ["현장 담당자", "관리자/책임자"]},
}

# ===== UI 표시용 메타데이터 (판정 로직에는 영향 없음) =====
# 등급별 브랜드 컬러 (강조색 / 옅은 배경 / 테두리)
GRADE_COLOR = {"정상": "#16a34a", "관찰": "#ca8a04", "주의": "#ea580c", "위험": "#dc2626"}
GRADE_BG    = {"정상": "#ecfdf3", "관찰": "#fefce8", "주의": "#fff7ed", "위험": "#fef2f2"}
GRADE_BORDER = {"정상": "#bbf7d0", "관찰": "#fde68a", "주의": "#fed7aa", "위험": "#fecaca"}

# 등급별 권장 조치 (상세 문구) / 등급별 한 줄 안내 (범례용)
GRADE_ACTION = {
    "정상": "이상 없음 — 별도 조치가 필요하지 않습니다.",
    "관찰": "정기 점검 시 확인해 주세요.",
    "주의": "24시간 내 현장 점검 계획 수립을 권장합니다.",
    "위험": "즉시 현장 점검이 필요합니다.",
}
GRADE_GUIDE = {
    "정상": "이상 없음",
    "관찰": "정기점검 시 확인",
    "주의": "점검 계획 수립",
    "위험": "즉시 점검 필요",
}

# 탐지 클래스 아이콘
CLASS_ICON = {"Damage": "⚠️", "Dirt": "🧹"}


def size_label(area_ratio):
    """박스 면적비율 → 사람이 읽기 쉬운 크기 등급 (표시 전용)"""
    if area_ratio < 0.01:
        return "Small"
    elif area_ratio < 0.03:
        return "Medium"
    return "Large"


def score_one(cls, area_ratio, conf):
    """객체 하나 점수 = 클래스가중치 × 크기가중치 × 신뢰도"""
    size_w = 1 + area_ratio * 5
    return CLASS_WEIGHT[cls] * size_w * conf


def grade_from_score(score):
    if score == 0:
        return "정상", "🟢"
    elif score < 15:
        return "관찰", "🟡"
    elif score < 40:
        return "주의", "🟠"
    else:
        return "위험", "🔴"


def assess(detections, turbine_id="터빈-01"):
    """
    detections: [{"cls":1, "area_ratio":0.2, "conf":0.9}, ...]
    반환: 판정 결과 dict
    """
    total = 0.0
    counted, recheck = [], []
    for d in detections:
        d = d.copy()
        d["label"] = d.get("label", CLASSES.get(d["cls"], str(d["cls"])))
        if d["conf"] < CONF_LOW:
            continue                      # 오탐 가능 → 무시
        if d["conf"] < CONF_HIGH:
            recheck.append(d)             # 재확인 플래그 (점수 미반영)
            continue
        total += score_one(d["cls"], d["area_ratio"], d["conf"])
        counted.append(d)

    grade, emoji = grade_from_score(total)
    esc = ESCALATION[grade]
    n_damage = sum(1 for d in counted if d["cls"] == 1)
    n_dirt   = sum(1 for d in counted if d["cls"] == 0)

    msg = f"{emoji} [{turbine_id}] {grade} (위험도 {total:.1f}) — Damage {n_damage}건 / Dirt {n_dirt}건"
    if recheck:
        msg += f" | ⚠️ 재확인 필요 {len(recheck)}건 (확신도 낮음, 사람 확인)"

    return {
        "turbine_id": turbine_id,
        "score": round(total, 1),
        "grade": grade, "emoji": emoji,
        "n_damage": n_damage, "n_dirt": n_dirt,
        "recheck": len(recheck),
        "channels": esc["channels"], "targets": esc["targets"],
        "message": msg,
        "counted": counted, "recheck_list": recheck,
    }


def detections_from_yolo(result):
    """
    YOLO predict 결과 1장 → detections 리스트로 변환.
    박스 면적비율 = (박스 넓이) / (이미지 넓이)
    """
    dets = []
    h, w = result.orig_shape           # 원본 이미지 크기
    img_area = h * w
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        # xyxy 픽셀 좌표
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / img_area
        dets.append({
            "cls": cls,
            "label": CLASSES.get(cls, str(cls)),
            "area_ratio": area_ratio,
            "conf": conf,
        })
    return dets


def assess_image(model, image_path, turbine_id="터빈-01", conf=0.25):
    """이미지 경로 → YOLO 추론 → 심각도 판정까지 한 번에"""
    result = model.predict(image_path, conf=conf, verbose=False)[0]
    dets = detections_from_yolo(result)
    assessment = assess(dets, turbine_id)
    assessment["yolo_result"] = result   # 박스 그린 이미지 등에 사용
    return assessment


def assess_video(
    model, video_path, turbine_id="터빈-01", conf=0.25,
    sample_interval_sec=1.0, max_frames=30,
):
    """
    동영상 경로 → 일정 간격으로 프레임을 샘플링해 각각 탐지하고,
    위험도가 가장 높은 프레임을 대표 결과로 반환한다.
    (정지 이미지 1장을 다루는 assess_image와 동일한 반환 형식 + 프레임 메타데이터)

    반환값의 "frame_results"에는 샘플링한 모든 프레임의 (초, YOLO 결과)가
    시간순으로 담기며, 탐지 결과를 타임랩스 영상으로 만들 때 사용한다.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"동영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(round(fps * sample_interval_sec)), 1)

    best = None  # (score, result, assessment, frame_index)
    frame_results = []  # [{"timestamp_sec": ..., "yolo_result": ...}, ...] 시간순
    frame_index = 0
    sampled = 0
    try:
        while sampled < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                result = model.predict(frame, conf=conf, verbose=False)[0]
                dets = detections_from_yolo(result)
                assessment = assess(dets, turbine_id)
                frame_results.append({
                    "timestamp_sec": round(frame_index / fps, 1),
                    "yolo_result": result,
                })
                if best is None or assessment["score"] > best[2]["score"]:
                    best = (result, frame_index, assessment)
                sampled += 1
            frame_index += 1
    finally:
        cap.release()

    if best is None:
        raise ValueError("동영상에서 프레임을 읽을 수 없습니다.")

    result, frame_index, assessment = best
    assessment["yolo_result"] = result
    assessment["source"] = "video"
    assessment["frame_index"] = frame_index
    assessment["timestamp_sec"] = round(frame_index / fps, 1)
    assessment["sampled_frames"] = sampled
    assessment["frame_results"] = frame_results
    return assessment


if __name__ == "__main__":
    # 모듈 단독 테스트 (YOLO 없이 로직만)
    cases = {
        "정상": [],
        "오염만": [{"cls":0,"area_ratio":0.03,"conf":0.8}],
        "큰 손상+오염": [
            {"cls":1,"area_ratio":0.25,"conf":0.92},
            {"cls":1,"area_ratio":0.15,"conf":0.78},
            {"cls":0,"area_ratio":0.05,"conf":0.7},
        ],
        "애매한 손상": [{"cls":1,"area_ratio":0.2,"conf":0.55}],
    }
    for name, dets in cases.items():
        r = assess(dets, "터빈-03")
        print(r["message"])
        print("   채널:", r["channels"], "| 대상:", r["targets"] or "없음")

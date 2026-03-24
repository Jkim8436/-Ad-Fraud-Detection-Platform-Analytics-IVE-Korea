import re
import base64
import json
import csv
import time
import math
import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from google import genai

# =========================================================
# 0) 페이지 기본 설정
# =========================================================
st.set_page_config(
    page_title="유사 광고 기반 매체 추천 대시보드",
    layout="wide"
)

# =========================================================
# 1) 경로 설정
#    대표님 환경에 맞게 여기만 수정하시면 됩니다.
# =========================================================
DOMAIN_FILE_MAP = {
    1: r"D:\내배캠 관련 모음\최종 프로젝트 데이터 셋\아이브 데이터 셋\분석에 사용할 전처리 완료 최종 데이터 셋\도메인별 분류 테이블\outputs_도메인 1_ads_idx단위 상위메인_최종 v4\domain1\domain1_mda_ads_detail_with_segment.csv",
    2: r"D:\내배캠 관련 모음\최종 프로젝트 데이터 셋\아이브 데이터 셋\분석에 사용할 전처리 완료 최종 데이터 셋\도메인별 분류 테이블\outputs_도메인 2_ads_idx단위 상위메인_최종 v4\domain2\domain2_mda_ads_detail_with_segment.csv",
    3: r"D:\내배캠 관련 모음\최종 프로젝트 데이터 셋\아이브 데이터 셋\분석에 사용할 전처리 완료 최종 데이터 셋\도메인별 분류 테이블\outputs_도메인 3_ads_idx단위 상위메인_최종 v4\domain3\domain3_mda_ads_detail_with_segment.csv",
    4: r"D:\내배캠 관련 모음\최종 프로젝트 데이터 셋\아이브 데이터 셋\분석에 사용할 전처리 완료 최종 데이터 셋\도메인별 분류 테이블\outputs_도메인 4_ads_idx단위 상위메인_최종 v4\domain4\domain4_mda_ads_detail_with_segment.csv",
    5: r"D:\내배캠 관련 모음\최종 프로젝트 데이터 셋\아이브 데이터 셋\분석에 사용할 전처리 완료 최종 데이터 셋\도메인별 분류 테이블\outputs_도메인 5_ads_idx단위 상위메인_최종 v4\domain5\domain5_mda_ads_detail_with_segment.csv",
}

MDA_MASTER_PATH = r"D:\내배캠 관련 모음\최종 프로젝트 데이터 셋\아이브 데이터 셋\분석에 사용할 전처리 완료 최종 데이터 셋\스트림밋 용 데이터 셋\tb_mda_master_ops v2.csv"

GEMINI_JSON_PATH = r"D:\내배캠 관련 모음\제미나이 api 키\gemini_config.json"
MODEL_ID = "gemini-2.0-flash"

SAMDASU_AI_ICON_PATH = r"C:\Users\LIM.J.S\Desktop\최종 프로젝트\삼다수 ai 사진.png"

# =========================================================
# 2) 라벨 매핑
# =========================================================
DOMAIN_LABEL_MAP = {
    "엔터테인먼트": 1,
    "금융": 2,
    "라이프스타일": 3,
    "커머스": 4,
    "기타": 5,
}
DOMAIN_LABEL_MAP_REV = {v: k for k, v in DOMAIN_LABEL_MAP.items()}

ADS_TYPE_LABEL_MAP = {
    1: "설치형",
    2: "실행형",
    3: "참여형",
    4: "클릭형",
    5: "페북",
    6: "트위터",
    7: "인스타",
    8: "노출형",
    9: "퀘스트",
    10: "유튜브",
    11: "네이버",
    12: "CPS(물건구매)",
}
ADS_TYPE_LABEL_MAP_REV = {v: k for k, v in ADS_TYPE_LABEL_MAP.items()}

# =========================================================
# 3) 컬럼명 설정
# =========================================================
NAME_COL = "ads_name"
ADS_IDX_COL = "ads_idx"
MDA_IDX_COL = "mda_idx"
ADS_TYPE_COL = "ads_type"
RPC_COL = "revenue_per_click"
MARGIN_COL = "margin_rate"

SIMILAR_LIST_COLS = [
    "ads_name", "ads_idx", "mda_idx",
    "clicks", "conversions", "cvr",
    "revenue_per_click", "margin_rate",
    "partner_payout", "reward_cost"
]

BEST_MDA_COLS = [
    "mda_idx", "profit", "clicks", "conversions", "cvr",
    "revenue_per_click", "margin_rate", "partner_payout"
]

# =========================================================
# 4) Gemini 설정
# =========================================================
with open(GEMINI_JSON_PATH, "r", encoding="utf-8") as json_file:
    config = json.load(json_file)

client = genai.Client(api_key=config["GEMINI_API_KEY"])

def make_answer(question: str) -> str:
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=question
    )
    return response.text

def fmt_seconds(sec: float) -> str:
    try:
        sec = float(sec)
    except Exception:
        return "N/A"
    if not math.isfinite(sec) or sec < 0:
        return "N/A"
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

# =========================================================
# 5) 운영 파라미터
# =========================================================
RUN_MODE = "prod"
TEST_N = 300
BATCH_SIZE = 40
MAX_CANDIDATES = None

BASE_THROTTLE_SEC = 0.0
MAX_RETRIES = 6
MAX_BACKOFF_SEC = 120
COOLDOWN_429_SEC = 30
MIN_SPLIT_SIZE = 6

# =========================================================
# 6) 전역 Rate Limiter
# =========================================================
GLOBAL_TARGET_RPM = 15
GLOBAL_MIN_INTERVAL_SEC = max(0.1, 60.0 / float(GLOBAL_TARGET_RPM))

class GlobalRateLimiter:
    def __init__(self, min_interval_sec: float):
        self.min_interval_sec = float(min_interval_sec)
        self._lock = threading.Lock()
        self._last_call_ts = 0.0

    def wait(self):
        with self._lock:
            now = time.time()
            if self._last_call_ts <= 0:
                self._last_call_ts = now
                return
            elapsed = now - self._last_call_ts
            if elapsed < self.min_interval_sec:
                time.sleep(self.min_interval_sec - elapsed)
            self._last_call_ts = time.time()

GLOBAL_RATE_LIMITER = GlobalRateLimiter(GLOBAL_MIN_INTERVAL_SEC)

# =========================================================
# 7) 로그 폴더
# =========================================================
OUT_BASE_DIR = Path("./similar_ads_logs")
OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

FAIL_RAW_DIR = OUT_BASE_DIR / "_llm_fail_raw"
FAIL_RAW_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = OUT_BASE_DIR / "_llm_batch_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
BATCH_LOG_JSONL = LOG_DIR / f"batch_log_similar_ads_{RUN_ID}.jsonl"
BATCH_LOG_CSV = LOG_DIR / f"batch_log_similar_ads_{RUN_ID}.csv"

CSV_FIELDS = [
    "run_id", "batch_id", "event", "attempt",
    "batch_size", "min_size", "split_depth",
    "status", "error_type", "error_message",
    "latency_sec", "cooldown_sec",
    "result_count", "raw_saved_path",
    "items_hash", "items_sample",
    "query_ad_name",
    "ts"
]

if not BATCH_LOG_CSV.exists():
    with open(BATCH_LOG_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()

def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _items_hash(items: List[Dict[str, Any]]) -> str:
    h = hashlib.sha1()
    for x in items:
        row = f"{x.get('row_id','')}|{x.get(NAME_COL,'')}|{x.get(ADS_IDX_COL,'')}|{x.get(MDA_IDX_COL,'')}\n"
        h.update(row.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]

def log_batch_row(row: Dict[str, Any]):
    row = dict(row)
    row["ts"] = row.get("ts", _now_ts())

    with open(BATCH_LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(BATCH_LOG_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writerow(row)

# =========================================================
# 8) 공통 유틸
# =========================================================
def sanitize_text_for_prompt(x: Any, max_len: int = 120) -> str:
    x = "" if x is None else str(x)
    x = x.replace("\n", " ").replace("\r", " ").strip()
    x = re.sub(r"\s+", " ", x)
    if len(x) > max_len:
        x = x[:max_len] + "…"
    return x

def extract_json_array(text: str) -> List[Any]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON array found in model output.")
    obj = json.loads(m.group(0))
    if not isinstance(obj, list):
        raise ValueError("Parsed JSON is not a list.")
    return obj

def is_rate_limit_429(msg: str) -> bool:
    if not msg:
        return False
    s = msg.lower()
    return ("429" in s) or ("resource_exhausted" in s)

def classify_error_type(err_msg: str) -> str:
    s = (err_msg or "").lower()
    if "batch output size mismatch" in s:
        return "mismatch"
    if is_rate_limit_429(s):
        return "rate_limit_429"
    if "no json array found" in s or "parsed json" in s or "json" in s:
        return "parse_error"
    return "other"

def dump_fail_raw(raw: str, prompt: str, items: List[Dict[str, Any]], err: Exception,
                  batch_id: int, split_depth: int, query_ad_name: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    f = FAIL_RAW_DIR / f"fail_{RUN_ID}_b{batch_id}_d{split_depth}_{ts}_n{len(items)}.json"
    payload = {
        "run_id": RUN_ID,
        "timestamp": ts,
        "batch_id": batch_id,
        "split_depth": split_depth,
        "query_ad_name": query_ad_name,
        "error": str(err),
        "item_count": len(items),
        "items_sample": items[:10],
        "prompt": prompt,
        "raw": raw,
    }
    f.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(f)

def safe_sort_numeric_like(values: List[Any]) -> List[Any]:
    def _key(x):
        try:
            return (0, float(x))
        except Exception:
            return (1, str(x))
    return sorted(values, key=_key)

def get_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# =========================================================
# 9) 데이터 로드
# =========================================================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def validate_columns(df: pd.DataFrame, required_cols: List[str], file_name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{file_name}에 필요한 컬럼이 없습니다: {missing}")

# =========================================================
# 10) LLM 후보 생성 / 프롬프트
# =========================================================
def build_candidate_items(df: pd.DataFrame) -> List[Dict[str, Any]]:
    use_cols = [NAME_COL, ADS_IDX_COL, MDA_IDX_COL]
    validate_columns(df, use_cols, "filtered_df")

    temp = df[use_cols].copy()
    temp[NAME_COL] = temp[NAME_COL].fillna("").astype(str)
    temp = temp[temp[NAME_COL].str.strip() != ""].copy()
    temp = temp.drop_duplicates(subset=[NAME_COL, ADS_IDX_COL, MDA_IDX_COL]).reset_index(drop=True)

    if MAX_CANDIDATES is not None:
        temp = temp.head(MAX_CANDIDATES).copy()

    items: List[Dict[str, Any]] = []
    for i, row in temp.iterrows():
        items.append({
            "row_id": int(i),
            NAME_COL: str(row[NAME_COL]),
            ADS_IDX_COL: row[ADS_IDX_COL],
            MDA_IDX_COL: row[MDA_IDX_COL],
        })
    return items

def build_similar_ads_prompt(query_ad_name: str, items: List[Dict[str, Any]]) -> str:
    query_ad_name = sanitize_text_for_prompt(query_ad_name, max_len=200)

    lines = []
    for x in items:
        lines.append(
            f'{x["row_id"]}: ads_name="{sanitize_text_for_prompt(x.get(NAME_COL, ""), 120)}", '
            f'ads_idx="{x.get(ADS_IDX_COL, "")}", '
            f'mda_idx="{x.get(MDA_IDX_COL, "")}"'
        )
    joined = "\n".join(lines)

    return f"""
당신은 광고명 의미 유사도 판별기입니다.

사용자 입력 광고명과, 아래 후보 광고 목록을 비교하여
\"광고의 성격/주제/서비스 목적/업종 맥락\"이 유사한 후보만 고르세요.

[중요 판정 기준]
- 단순히 단어 몇 개가 겹친다고 유사하다고 판단하지 마세요.
- 광고명에 담긴 실제 서비스 맥락이 유사해야 합니다.
- 예:
  - 게임 사전예약 ↔ 다른 게임 사전예약 : 유사
  - 금융 앱 이벤트 ↔ 쇼핑몰 쿠폰 이벤트 : 비유사
  - 웹툰 쿠키 충전 ↔ 콘텐츠 감상형 리워드 : 유사 가능
  - 같은 \"이벤트\", \"혜택\", \"적립\" 단어가 있어도 업종/서비스 맥락이 다르면 제외
- query 광고와 후보 광고가 같은 산업/서비스 맥락이면 포함하세요.
- 애매하면 제외하세요. (억지 포함 금지)

[사용자 입력 광고명]
\"{query_ad_name}\"

[후보 광고 목록]
{joined}

[출력 규칙 - 매우 중요]
- 반드시 JSON 배열만 출력하세요.
- 다른 설명/문장/코드블록/주석 절대 금지
- 배열 원소는 반드시 정수 row_id만 허용
- 유사한 후보가 없으면 빈 배열 [] 만 출력
- 중복 없이 출력
- 예시: [0, 3, 7]

이제 유사한 후보의 row_id 배열만 출력하세요.
""".strip()

# =========================================================
# 11) LLM 호출 / 재시도 / 자동분할
# =========================================================
def gemini_call_once(
    query_ad_name: str,
    items: List[Dict[str, Any]],
    make_answer_fn,
    batch_id: int,
    attempt: int,
    split_depth: int
) -> List[int]:

    prompt = build_similar_ads_prompt(query_ad_name, items)
    items_h = _items_hash(items)
    sample = " | ".join([str(x.get(NAME_COL, "")) for x in items[:5]])

    t0 = time.time()
    raw = ""
    cooldown = 0

    GLOBAL_RATE_LIMITER.wait()

    try:
        raw = make_answer_fn(prompt)
        arr = extract_json_array(raw)

        row_id_set = {int(x["row_id"]) for x in items}
        out: List[int] = []

        for i, v in enumerate(arr):
            if isinstance(v, str) and v.strip().isdigit():
                v = int(v.strip())
            if not isinstance(v, int):
                raise ValueError(f"Invalid row_id type at index {i}: {v}")
            if v not in row_id_set:
                raise ValueError(f"Invalid row_id out of candidate range: {v}")
            out.append(v)

        out = sorted(set(out))

        latency = time.time() - t0
        log_batch_row({
            "run_id": RUN_ID,
            "batch_id": batch_id,
            "event": "call",
            "attempt": attempt,
            "batch_size": len(items),
            "min_size": MIN_SPLIT_SIZE,
            "split_depth": split_depth,
            "status": "success",
            "error_type": "",
            "error_message": "",
            "latency_sec": round(latency, 3),
            "cooldown_sec": cooldown,
            "result_count": len(out),
            "raw_saved_path": "",
            "items_hash": items_h,
            "items_sample": sample,
            "query_ad_name": query_ad_name,
        })
        return out

    except Exception as e:
        latency = time.time() - t0
        msg = str(e)
        etype = classify_error_type(msg)

        raw_path = dump_fail_raw(
            raw=raw,
            prompt=prompt,
            items=items,
            err=e,
            batch_id=batch_id,
            split_depth=split_depth,
            query_ad_name=query_ad_name
        )

        if etype == "rate_limit_429":
            cooldown = COOLDOWN_429_SEC
            log_batch_row({
                "run_id": RUN_ID,
                "batch_id": batch_id,
                "event": "cooldown",
                "attempt": attempt,
                "batch_size": len(items),
                "min_size": MIN_SPLIT_SIZE,
                "split_depth": split_depth,
                "status": "cooldown_applied",
                "error_type": "rate_limit_429",
                "error_message": msg[:240],
                "latency_sec": round(latency, 3),
                "cooldown_sec": cooldown,
                "result_count": 0,
                "raw_saved_path": raw_path,
                "items_hash": items_h,
                "items_sample": sample,
                "query_ad_name": query_ad_name,
            })
            time.sleep(cooldown)

        log_batch_row({
            "run_id": RUN_ID,
            "batch_id": batch_id,
            "event": "call",
            "attempt": attempt,
            "batch_size": len(items),
            "min_size": MIN_SPLIT_SIZE,
            "split_depth": split_depth,
            "status": "fail",
            "error_type": etype,
            "error_message": msg[:240],
            "latency_sec": round(latency, 3),
            "cooldown_sec": cooldown,
            "result_count": 0,
            "raw_saved_path": raw_path,
            "items_hash": items_h,
            "items_sample": sample,
            "query_ad_name": query_ad_name,
        })
        raise

def select_similar_with_split(
    query_ad_name: str,
    items: List[Dict[str, Any]],
    make_answer_fn,
    batch_id: int,
    split_depth: int = 0
) -> List[int]:

    if len(items) == 0:
        return []

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return gemini_call_once(
                query_ad_name=query_ad_name,
                items=items,
                make_answer_fn=make_answer_fn,
                batch_id=batch_id,
                attempt=attempt,
                split_depth=split_depth,
            )
        except Exception as e:
            msg = str(e)
            etype = classify_error_type(msg)

            if etype == "mismatch":
                if len(items) <= MIN_SPLIT_SIZE:
                    log_batch_row({
                        "run_id": RUN_ID,
                        "batch_id": batch_id,
                        "event": "fallback",
                        "attempt": attempt,
                        "batch_size": len(items),
                        "min_size": MIN_SPLIT_SIZE,
                        "split_depth": split_depth,
                        "status": "fallback_to_empty",
                        "error_type": "mismatch",
                        "error_message": msg[:240],
                        "latency_sec": 0,
                        "cooldown_sec": 0,
                        "result_count": 0,
                        "raw_saved_path": "",
                        "items_hash": _items_hash(items),
                        "items_sample": " | ".join([str(x.get(NAME_COL, "")) for x in items[:5]]),
                        "query_ad_name": query_ad_name,
                    })
                    return []

                mid = len(items) // 2
                left = items[:mid]
                right = items[mid:]

                log_batch_row({
                    "run_id": RUN_ID,
                    "batch_id": batch_id,
                    "event": "split",
                    "attempt": attempt,
                    "batch_size": len(items),
                    "min_size": MIN_SPLIT_SIZE,
                    "split_depth": split_depth,
                    "status": "split_into_two",
                    "error_type": "mismatch",
                    "error_message": msg[:240],
                    "latency_sec": 0,
                    "cooldown_sec": 0,
                    "result_count": 0,
                    "raw_saved_path": "",
                    "items_hash": _items_hash(items),
                    "items_sample": " | ".join([str(x.get(NAME_COL, "")) for x in items[:5]]),
                    "query_ad_name": query_ad_name,
                })

                out_left = select_similar_with_split(
                    query_ad_name=query_ad_name,
                    items=left,
                    make_answer_fn=make_answer_fn,
                    batch_id=batch_id,
                    split_depth=split_depth + 1
                )
                out_right = select_similar_with_split(
                    query_ad_name=query_ad_name,
                    items=right,
                    make_answer_fn=make_answer_fn,
                    batch_id=batch_id,
                    split_depth=split_depth + 1
                )
                return sorted(set(out_left + out_right))

            backoff = min(MAX_BACKOFF_SEC, (attempt * 0.5))
            log_batch_row({
                "run_id": RUN_ID,
                "batch_id": batch_id,
                "event": "retry_wait",
                "attempt": attempt,
                "batch_size": len(items),
                "min_size": MIN_SPLIT_SIZE,
                "split_depth": split_depth,
                "status": "waiting",
                "error_type": etype,
                "error_message": msg[:240],
                "latency_sec": 0,
                "cooldown_sec": backoff,
                "result_count": 0,
                "raw_saved_path": "",
                "items_hash": _items_hash(items),
                "items_sample": " | ".join([str(x.get(NAME_COL, "")) for x in items[:5]]),
                "query_ad_name": query_ad_name,
            })
            time.sleep(backoff)

    log_batch_row({
        "run_id": RUN_ID,
        "batch_id": batch_id,
        "event": "fallback",
        "attempt": MAX_RETRIES + 1,
        "batch_size": len(items),
        "min_size": MIN_SPLIT_SIZE,
        "split_depth": split_depth,
        "status": "fallback_to_empty",
        "error_type": "final_fail",
        "error_message": "",
        "latency_sec": 0,
        "cooldown_sec": 0,
        "result_count": 0,
        "raw_saved_path": "",
        "items_hash": _items_hash(items),
        "items_sample": " | ".join([str(x.get(NAME_COL, "")) for x in items[:5]]),
        "query_ad_name": query_ad_name,
    })
    return []

# =========================================================
# 12) 유사 광고 탐색
# =========================================================
def find_similar_ads(
    filtered_df: pd.DataFrame,
    query_ad_name: str,
    make_answer_fn,
    batch_size: int = BATCH_SIZE
) -> pd.DataFrame:

    if not isinstance(filtered_df, pd.DataFrame):
        raise ValueError("filtered_df는 pandas DataFrame이어야 합니다.")

    if not str(query_ad_name).strip():
        raise ValueError("query_ad_name이 비어 있습니다.")

    df = filtered_df.copy()

    if RUN_MODE == "test":
        df = df.head(TEST_N).copy()

    candidate_items = build_candidate_items(df)

    if len(candidate_items) == 0:
        return pd.DataFrame(columns=[NAME_COL, ADS_IDX_COL, MDA_IDX_COL])

    all_selected_row_ids: List[int] = []
    processed = 0
    t0 = time.time()
    batch_id = 0

    progress_area = st.empty()

    for b_start in range(0, len(candidate_items), batch_size):
        batch_items = candidate_items[b_start:b_start + batch_size]

        selected_ids = select_similar_with_split(
            query_ad_name=query_ad_name,
            items=batch_items,
            make_answer_fn=make_answer_fn,
            batch_id=batch_id,
            split_depth=0
        )

        all_selected_row_ids.extend(selected_ids)

        processed += len(batch_items)
        batch_id += 1

        elapsed = time.time() - t0
        remain = len(candidate_items) - processed
        speed = processed / elapsed if elapsed > 0 else 0.0
        eta_sec = remain / speed if speed > 0 else float("inf")

        progress_area.info(
            f"유사 광고 분석 중... {processed}/{len(candidate_items)} | "
            f"속도: {speed:.2f} items/s | ETA: {fmt_seconds(eta_sec)}"
        )

        if BASE_THROTTLE_SEC and BASE_THROTTLE_SEC > 0:
            time.sleep(BASE_THROTTLE_SEC)

    result_items = [x for x in candidate_items if int(x["row_id"]) in set(all_selected_row_ids)]

    result_df = pd.DataFrame(result_items)
    if result_df.empty:
        progress_area.empty()
        return pd.DataFrame(columns=[NAME_COL, ADS_IDX_COL, MDA_IDX_COL])

    result_df = result_df[[NAME_COL, ADS_IDX_COL, MDA_IDX_COL]].drop_duplicates().reset_index(drop=True)

    progress_area.empty()
    return result_df

def filter_original_by_similar_ads_idx(
    filtered_df: pd.DataFrame,
    similar_ads_df: pd.DataFrame
) -> pd.DataFrame:
    ads_idx_values = pd.Series(similar_ads_df[ADS_IDX_COL]).dropna().unique().tolist()
    out = filtered_df[filtered_df[ADS_IDX_COL].isin(ads_idx_values)].copy()
    return out

# =========================================================
# 13) 데이터 전처리 보조
# =========================================================
def normalize_action_type(x: Any) -> str:
    x = "" if x is None else str(x).strip()
    x = re.sub(r"\s+", "", x)
    return x

def get_existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def format_stat_value(x: Any, digits: int = 4) -> str:
    if pd.isna(x):
        return "N/A"
    x = float(x)
    if abs(x - round(x)) < 1e-9:
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}".rstrip("0").rstrip(".")


def format_stat_value_1f(x: Any) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{float(x):,.1f}"

def format_percent_1f(x: Any) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{float(x) * 100:.1f}%"


def build_domain_ads_top_media(
    mda_master_df: pd.DataFrame,
    selected_domain_label: int,
    selected_ads_type_value: int,
    top_n: int = 10
) -> pd.DataFrame:
    df = mda_master_df[
        (pd.to_numeric(mda_master_df["domain_label"], errors="coerce") == selected_domain_label) &
        (pd.to_numeric(mda_master_df["ads_type"], errors="coerce") == selected_ads_type_value)
    ].copy()

    if df.empty:
        return df

    sort_col = "conversions"

    df = (
        df.sort_values(by=sort_col, ascending=False)
        .drop_duplicates(subset=[MDA_IDX_COL])
        .head(top_n)
        .copy()
    )
    return df

def summarize_top_media(df: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
    if df.empty:
        return {
            "mda_ids": [],
            "mda_ids_text": "없음",
            "rpc_mean": None,
            "rpc_median": None,
            "margin_mean": None,
            "margin_median": None,
            "cvr_mean": None,
        }

    top_df = df.head(top_n).copy()

    rpc = pd.to_numeric(top_df[RPC_COL], errors="coerce")
    margin = pd.to_numeric(top_df[MARGIN_COL], errors="coerce")
    cvr = pd.to_numeric(top_df["cvr"], errors="coerce")

    mda_ids = top_df[MDA_IDX_COL].astype(str).tolist()

    return {
        "mda_ids": mda_ids,
        "mda_ids_text": ", ".join(mda_ids),
        "rpc_mean": rpc.mean(),
        "rpc_median": rpc.median(),
        "margin_mean": margin.mean(),
        "margin_median": margin.median(),
        "cvr_mean": cvr.mean(),
    }

def build_insight_text(
    result_status: str,
    query_ad_name: str,
    selected_domain_name: str,
    selected_ads_type_label: str,
    expand_df: pd.DataFrame,
    keep_df: pd.DataFrame
) -> str:
    query_ad_name = str(query_ad_name).strip()
    selected_domain_name = str(selected_domain_name).strip()
    selected_ads_type_label = str(selected_ads_type_label).strip()


    if result_status == "expand_exists":
        s = summarize_top_media(expand_df, top_n=5)
        return (
            f'[<b>{query_ad_name}</b>] 광고의 유사 광고 분석 결과입니다.<br><br>'
            f'우선 검토할 <b>최적 운영 매체 ID</b>는 <b>{s["mda_ids_text"]}</b>입니다.<br><br>'
            f'<b>클릭당 광고주 단가</b>: 평균 {format_stat_value_1f(s["rpc_mean"])}원 / '
            f'중앙값 {format_stat_value_1f(s["rpc_median"])}원<br>'
            f'<b>클릭당 아이브 마진율</b>: 평균 {format_percent_1f(s["margin_mean"])} / '
            f'중앙값 {format_percent_1f(s["margin_median"])}<br>'
            f'<b>평균 전환율</b>: {format_percent_1f(s["cvr_mean"])}<br><br>'
            f'전반적으로 수익성과 전환 효율이 함께 확보된 매체군으로 판단됩니다.<br>'
            f'자세한 정보는 아래 <b>우선 추천 매체 top 10</b> 테이블을 확인하세요.'
        )

    if result_status == "keep_only":
        s = summarize_top_media(keep_df, top_n=5)
        return (
            f'[<b>{query_ad_name}</b>] 광고의 유사 광고 분석 결과입니다.<br><br>'
            f'우선 추천 매체는 확인되지 않았습니다.<br>'
            f'대신 차선책으로 검토할 <b>추가 추천 매체 후보 ID</b>는 <b>{s["mda_ids_text"]}</b>입니다.<br><br>'
            f'<b>클릭당 광고주 단가</b>: 평균 {format_stat_value_1f(s["rpc_mean"])}원 / '
            f'중앙값 {format_stat_value_1f(s["rpc_median"])}원<br>'
            f'<b>클릭당 아이브 마진율</b>: 평균 {format_percent_1f(s["margin_mean"])} / '
            f'중앙값 {format_percent_1f(s["margin_median"])}<br>'
            f'<b>평균 전환율</b>: {format_percent_1f(s["cvr_mean"])}<br><br>'
            f'다만 우선 추천 매체 대비 규모나 효율 측면에서 보수적인 검토가 필요합니다.'
            f'자세한 정보는 아래 <b>추가 추천 매체 후보 top 10</b> 테이블을 확인하세요.'
        )

    if result_status == "filter_no_match":
        return (
            f'<br>선택하신 단가 조건에 맞는 매체가 확인되지 않으며,<br> '
            f'해당 단가 조건의 매체는 현재 <b>개선/축소가 필요한 매체</b>일 가능성이 높습니다.<br> '
            f'단가 조건을 조정해주세요.'
        )

    if result_status == "no_recommendation":
        return (
            f'[{query_ad_name}] 광고의 유사 광고 분석 결과, '
            f'현재 유사 광고 기반으로는 추천 가능한 운영 매체를 확인하지 못했습니다.<br>'
            f'이는 관련 매체가 주로 개선 또는 축소가 필요한 상태이기 때문입니다.<br>'
            f'차선책으로 검색하신 <b>{selected_domain_name}</b>의 '
            f'<b>{selected_ads_type_label}</b> 내 상위 매체를 함께 제공합니다.'
        )

    if result_status == "no_similar_ads":
        return (
            f'검색하신 <b>{query_ad_name}</b> 광고는 아이브 코리아 운영 데이터 내에서 '
            f'유사 광고가 확인되지 않았습니다.<br>'
            f'따라서 유사 광고 기반 추천 대신, 검색하신 '
            f'<b>{selected_domain_name}</b>의 <b>{selected_ads_type_label}</b> 내 상위 매체를 차선책으로 제공합니다.'
        )

    return "분석 결과 인사이트를 생성할 수 없습니다."

def render_insight_box(
    insight_html: str,
    icon_base64: str,
    margin_top_px: int = 14,
    margin_bottom_px: int = 18
):
    st.markdown(
        f"""<div id="insight-section" style="margin-top:{margin_top_px}px; margin-bottom:{margin_bottom_px}px;">
<div style="display:flex; align-items:flex-start; gap:18px;">
<div style="width:64px; min-width:64px; height:64px; border-radius:18px; background:#ffffff; border:1px solid #dbe4ff; display:flex; align-items:center; justify-content:center; overflow:hidden; box-shadow:0 4px 10px rgba(0,0,0,0.05); margin-top:10px;">
<img src="data:image/png;base64,{icon_base64}" style="width:48px; height:48px; object-fit:contain; display:block;" />
</div>
<div style="position:relative; flex:1; padding:20px 24px; border-radius:20px; background:#eef4ff; border:1px solid #dbe4ff; box-shadow:0 4px 12px rgba(0,0,0,0.04);">
<div style="position:absolute; left:-12px; top:26px; width:0; height:0; border-top:10px solid transparent; border-bottom:10px solid transparent; border-right:12px solid #eef4ff;"></div>
<div style="font-size:16px; font-weight:800; color:#1e3a8a; margin-bottom:10px;">분석 인사이트</div>
<div style="font-size:17px; line-height:1.8; font-weight:500; color:#1f2937;">{insight_html}</div>
</div>
</div>
</div>""",
        unsafe_allow_html=True
    )

def prepare_recommendation_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "partner_payout" in out.columns and "clicks" in out.columns:
        payout = pd.to_numeric(out["partner_payout"], errors="coerce")
        clicks = pd.to_numeric(out["clicks"], errors="coerce")
        out["partner_payout"] = payout.div(clicks.replace(0, pd.NA))

    return out


def apply_numeric_range_filter(df: pd.DataFrame, col: str, min_val: float, max_val: float) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df.copy()

    s = pd.to_numeric(df[col], errors="coerce")
    return df[s.between(min_val, max_val, inclusive="both")].copy()


def apply_text_contains_filter(df: pd.DataFrame, col: str, keyword: str) -> pd.DataFrame:
    if df.empty or col not in df.columns or not str(keyword).strip():
        return df.copy()

    keyword = str(keyword).strip()
    s = df[col].fillna("").astype(str)
    return df[s.str.contains(keyword, case=False, na=False)].copy()


def get_slider_min_max(df: pd.DataFrame, col: str, fallback_min: float = 0.0, fallback_max: float = 1.0):
    if df.empty or col not in df.columns:
        return fallback_min, fallback_max

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return fallback_min, fallback_max

    min_val = float(s.min())
    max_val = float(s.max())

    if min_val == max_val:
        max_val = min_val + 0.0001

    return min_val, max_val

def rename_recommendation_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "mda_idx": "매체사 ID",
        "profit": "아이브 수익",
        "clicks": "클릭수",
        "conversions": "전환수",
        "cvr": "전환율",
        "revenue_per_click": "클릭당 광고주 단가",
        "margin_rate": "클릭당 아이브 마진율",
        "partner_payout": "매체사 지급 비용"
    }
    return df.rename(columns=rename_map)


def rename_similar_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "ads_name": "광고명",
        "ads_idx": "광고 ID",
        "mda_idx": "매체사 ID",
        "clicks": "클릭수",
        "conversions": "전환수",
        "cvr": "전환율",
        "revenue_per_click": "클릭당 광고주 단가",
        "margin_rate": "클릭당 아이브 마진율",
        "partner_payout": "매체사 지급 비용",
        "reward_cost": "리워드 비용"
    }
    return df.rename(columns=rename_map)


def init_recommendation_state():
    defaults = {
        "rec_query_ad_name": "",
        "rec_selected_domain_name": "엔터테인먼트",
        "rec_selected_ads_type_label": None,
        "rec_selected_rpc": "선택 안함",
        "rec_selected_margin": "선택 안함",
        "rec_analysis_done": False,
        "rec_is_running": False,
        "rec_similar_ads_df": pd.DataFrame(),
        "rec_final_similar_df": pd.DataFrame(),
        "rec_optimal_mda_df": pd.DataFrame(),
        "rec_fallback_top_df": pd.DataFrame(),
        "rec_result_status": "",
        "rec_match_ads_name_keyword": "",
        "rec_match_ads_idx_keyword": "",
        "rec_match_mda_idx_keyword": "",
        "rec_meta": {}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =========================================================
# 14) 페이지 렌더 함수
# =========================================================
def render_recommendation_page():
    init_recommendation_state()

    st.title("운영 매체 추천 시스템")

    st.markdown(
        """
        <style>
        div[data-testid="stDataFrame"] thead tr th {
            background-color: #fde8e8 !important;
            color: #7f1d1d !important;
        }

        .toc-link {
            display:block;
            padding:10px 14px;
            margin-bottom:8px;
            border:1px solid #d1d5db;
            border-radius:12px;
            background:#ffffff;
            color:#1f2937 !important;
            font-weight:700;
            text-decoration:none !important;
            text-align:center;
        }

        .toc-link:hover {
            background:#f9fafb;
            border-color:#9ca3af;
        }

        #search-section,
        #insight-section,
        #recommend-section,
        #similar-section {
            scroll-margin-top: 120px;
        }

        .filter-panel-wrap {
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 28px 28px 26px 28px;
            margin-top: 8px;
            margin-bottom: 30px;
        }

        .filter-panel-title {
            font-size: 18px;
            font-weight: 800;
            color: #111827;
            margin-bottom: 34px;
        }


        .st-key-filter_panel {
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 28px 28px 26px 28px;
            margin-top: 8px;
            margin-bottom: 30px;
        }

        .filter-panel-title {
            font-size: 18px;
            font-weight: 800;
            color: #111827;
            margin-bottom: 34px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )




    # -------------------------------------------------
    # 1) 먼저 현재 선택 상태 기준 도메인명 확보
    # -------------------------------------------------
    selected_domain_name = st.session_state.rec_selected_domain_name
    selected_domain_label = DOMAIN_LABEL_MAP[selected_domain_name]
    selected_domain_file = DOMAIN_FILE_MAP[selected_domain_label]

    try:
        domain_df = load_csv(selected_domain_file)
    except Exception as e:
        st.error(f"도메인 파일을 불러오지 못했습니다.\n{e}")
        st.stop()

    required_domain_cols = [
        NAME_COL, ADS_IDX_COL, MDA_IDX_COL,
        ADS_TYPE_COL, RPC_COL, MARGIN_COL
    ]
    try:
        validate_columns(domain_df, required_domain_cols, "도메인 파일")
    except Exception as e:
        st.error(str(e))
        st.stop()

    ads_type_values_raw = (
        pd.to_numeric(domain_df[ADS_TYPE_COL], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    ads_type_values = safe_sort_numeric_like(ads_type_values_raw)
    ads_type_label_options = [ADS_TYPE_LABEL_MAP.get(v, f"기타({v})") for v in ads_type_values]

    if (
        st.session_state.rec_selected_ads_type_label is None
        or st.session_state.rec_selected_ads_type_label not in ads_type_label_options
    ):
        st.session_state.rec_selected_ads_type_label = ads_type_label_options[0] if ads_type_label_options else None

    

    # -------------------------------------------------
    # 안내 카드
    # -------------------------------------------------
    samdasu_ai_icon_base64 = get_image_base64(SAMDASU_AI_ICON_PATH)

    st.markdown(
        f"""
        <style>
        .guide-card-c {{
            margin-top: 14px;
            margin-bottom: 40px;
            max-width: 1120px;
            background: #fce1e1;
            border: 1px solid #e5e7eb;
            border-radius: 24px;
            padding: 24px 28px;
            display: flex;
            align-items: center;
            gap: 40px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.04);
        }}

        .guide-card-icon-c {{
            width: 64px;
            height: 64px;
            min-width: 64px;
            border-radius: 18px;
            background: #ffffff;
            border: 1px solid #dbe4ff;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            overflow: hidden;
        }}

        .guide-card-icon-c img {{
            width: 48px;
            height: 48px;
            object-fit: contain;
            display: block;
        }}

        .guide-card-title-c {{
            font-size: 16px;
            font-weight: 800;
            color: #4b5563;
            margin-bottom: 6px;
            letter-spacing: 0.2px;
        }}

        .guide-card-text-c {{
            font-size: 18px;
            line-height: 1.7;
            font-weight: 500;
            color: #1f2937;
        }}

        .candidate-kpi-card {{
            height: 170px;
            border-radius: 20px;
            background: #f3f4f6;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin-top: 10px;
        }}

        .candidate-kpi-title {{
            font-size: 18px;
            font-weight: 700;
            color: #374151;
            margin-bottom: 12px;
            text-align: center;
        }}

        .candidate-kpi-value {{
            font-size: 34px;
            font-weight: 800;
            color: #111827;
            line-height: 1.1;
            text-align: center;
        }}

        div.stButton > button[kind="primary"] {{
            height: 170px;
            font-size: 24px;
            font-weight: 700;
            border-radius: 20px;
            margin-top: 10px;
        }}
        </style>

        <div class="guide-card-c">
            <div class="guide-card-icon-c">
                <img src="data:image/png;base64,{samdasu_ai_icon_base64}" />
            </div>
            <div>
                <div class="guide-card-title-c">AI 추천 안내</div>
                <div class="guide-card-text-c">
                    아래의 검색 조건에 맞게, 신규 광고명 및 기존 운영 광고명을 입력해주시면<br>
                    아이브 코리아 운영 데이터를 기반으로 유사한 광고들만 추출하여 최적의 운영 매체를 추천해드립니다
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------
    # 2) 검색 조건 UI
    # -------------------------------------------------
    
    st.markdown('<div id="search-section"></div>', unsafe_allow_html=True)
    st.subheader("검색 조건")

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    left_area, gap_area, right_area = st.columns([1.0, 0.12, 1.0])

    with left_area:
        query_ad_name = st.text_input(
            "광고명",
            placeholder="예: 프로야구 H2 사전예약",
            key="rec_query_ad_name"
        )

        row_1_col_1, row_1_col_2 = st.columns(2)
        with row_1_col_1:
            selected_domain_name = st.selectbox(
                "도메인",
                options=list(DOMAIN_LABEL_MAP.keys()),
                key="rec_selected_domain_name"
            )

        with row_1_col_2:
            selected_ads_type_label = st.selectbox(
                "광고 유형",
                options=ads_type_label_options,
                key="rec_selected_ads_type_label"
            )

        # row_2_col_1, row_2_col_2 = st.columns(2)
        # with row_2_col_1:
        #     selected_rpc = st.selectbox(
        #         "클릭당 광고주 단가",
        #         options=rpc_options,
        #         key="rec_selected_rpc"
        #     )

        # with row_2_col_2:
        #     selected_margin = st.selectbox(
        #         "아이브 마진율",
        #         options=margin_options,
        #         key="rec_selected_margin"
        #     )

    # 현재 UI에서 선택한 값 기준으로 다시 계산
    selected_domain_label = DOMAIN_LABEL_MAP[selected_domain_name]
    selected_ads_type_value = ADS_TYPE_LABEL_MAP_REV[selected_ads_type_label]

    filtered_df = domain_df[
        pd.to_numeric(domain_df[ADS_TYPE_COL], errors="coerce") == selected_ads_type_value
    ].copy()

    with gap_area:
        st.empty()

    with right_area:
        info_col, btn_col = st.columns([1, 0.5])

        with info_col:
            st.markdown(
                f"""
                <div class="candidate-kpi-card">
                    <div class="candidate-kpi-title">현재 조건의 후보 데이터 건수</div>
                    <div class="candidate-kpi-value">{len(filtered_df):,}건</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with btn_col:
            run_btn = st.button(
                "유사 광고 분석 실행",
                type="primary",
                use_container_width=True
            )

        

        # 버튼 바로 아래에 상태 박스를 고정해서 그릴 자리
        analysis_status_placeholder = st.empty()

        


    # -------------------------------------------------
    # 3) 버튼 클릭 시 분석 실행
    # -------------------------------------------------
    if run_btn:
        st.session_state.rec_analysis_done = False
        st.session_state.rec_result_status = ""

        if not str(query_ad_name).strip():
            st.warning("광고명을 입력해주셔야 합니다.")
        elif filtered_df.empty:
            st.warning("현재 조건에 해당하는 후보 데이터가 없습니다.")
        else:
            st.session_state.rec_is_running = True

            try:
                with st.spinner("Gemini로 유사 광고를 분석하고 있습니다... 분석 중에는 페이지를 이동하지 말아주세요."):
                    similar_ads_df = find_similar_ads(
                        filtered_df=filtered_df,
                        query_ad_name=query_ad_name,
                        make_answer_fn=make_answer
                    )

                    try:
                        mda_master_df = load_csv(MDA_MASTER_PATH)
                    except Exception as e:
                        st.error(f"tb_mda_master_ops v2.csv 파일을 불러오지 못했습니다.\n{e}")
                        st.stop()

                    if MDA_IDX_COL not in mda_master_df.columns:
                        st.error(f"tb_mda_master_ops v2.csv 에 {MDA_IDX_COL} 컬럼이 없습니다.")
                        st.stop()

                    if "action_type" not in mda_master_df.columns:
                        st.error("tb_mda_master_ops v2.csv 에 action_type 컬럼이 없습니다.")
                        st.stop()

                    fallback_top_df = build_domain_ads_top_media(
                        mda_master_df=mda_master_df,
                        selected_domain_label=selected_domain_label,
                        selected_ads_type_value=selected_ads_type_value,
                        top_n=10
                    )

                    if similar_ads_df.empty:
                        st.session_state.rec_analysis_done = True
                        st.session_state.rec_result_status = "no_similar_ads"
                        st.session_state.rec_similar_ads_df = pd.DataFrame(columns=[NAME_COL, ADS_IDX_COL, MDA_IDX_COL])
                        st.session_state.rec_final_similar_df = pd.DataFrame(columns=SIMILAR_LIST_COLS)
                        st.session_state.rec_optimal_mda_df = pd.DataFrame()
                        st.session_state.rec_fallback_top_df = fallback_top_df.copy()
                        st.session_state.rec_meta = {
                            "selected_domain_name": selected_domain_name,
                            "selected_domain_label": selected_domain_label,
                            "selected_ads_type_label": selected_ads_type_label,
                            "selected_ads_type_value": selected_ads_type_value,
                            "query_ad_name": query_ad_name,
                            "n_similar_ads": 0,
                            "n_final_rows": 0,
                            "n_optimal_rows": 0,
                        }

                    else:
                        final_similar_df = filter_original_by_similar_ads_idx(
                            filtered_df=filtered_df,
                            similar_ads_df=similar_ads_df
                        )

                        used_mda_list = pd.Series(similar_ads_df[MDA_IDX_COL]).dropna().unique().tolist()

                        mda_master_filtered_df = mda_master_df[
                            (pd.to_numeric(mda_master_df["domain_label"], errors="coerce") == selected_domain_label) &
                            (pd.to_numeric(mda_master_df["ads_type"], errors="coerce") == selected_ads_type_value)
                        ].copy()

                        optimal_mda_df = mda_master_filtered_df[
                            mda_master_filtered_df[MDA_IDX_COL].isin(used_mda_list)
                        ].copy()

                        optimal_mda_df["action_type_normalized"] = optimal_mda_df["action_type"].map(normalize_action_type)

                        keep_actions = {"확대", "유지"}
                        optimal_mda_df = optimal_mda_df[
                            optimal_mda_df["action_type_normalized"].isin(keep_actions)
                        ].copy()

                        expand_raw_df = optimal_mda_df[
                            optimal_mda_df["action_type_normalized"] == "확대"
                        ].copy()

                        keep_raw_df = optimal_mda_df[
                            optimal_mda_df["action_type_normalized"] == "유지"
                        ].copy()

                        if not expand_raw_df.empty:
                            result_status = "expand_exists"
                        elif not keep_raw_df.empty:
                            result_status = "keep_only"
                        else:
                            result_status = "no_recommendation"

                        st.session_state.rec_analysis_done = True
                        st.session_state.rec_result_status = result_status
                        st.session_state.rec_similar_ads_df = similar_ads_df.copy()
                        st.session_state.rec_final_similar_df = final_similar_df.copy()
                        st.session_state.rec_optimal_mda_df = optimal_mda_df.copy()
                        st.session_state.rec_fallback_top_df = fallback_top_df.copy()
                        st.session_state.rec_meta = {
                            "selected_domain_name": selected_domain_name,
                            "selected_domain_label": selected_domain_label,
                            "selected_ads_type_label": selected_ads_type_label,
                            "selected_ads_type_value": selected_ads_type_value,
                            # "selected_rpc": selected_rpc,
                            # "selected_margin": selected_margin,
                            "query_ad_name": query_ad_name,
                            "n_similar_ads": len(similar_ads_df),
                            "n_final_rows": len(final_similar_df),
                            "n_optimal_rows": len(optimal_mda_df),
                        }

            except Exception as e:
                st.error(f"유사 광고 분석 중 오류가 발생했습니다.\n{e}")
                st.stop()

            finally:
                st.session_state.rec_is_running = False

        
        
        # -------------------------------------------------
    # 3-1) 분석 상태 박스 렌더링
    # -------------------------------------------------
    if st.session_state.rec_analysis_done:
        n_similar_ads = st.session_state.rec_meta.get("n_similar_ads", 0)

        analysis_status_placeholder.markdown(
            f"""
            <div style="
                padding:14px 18px;
                border-radius:14px;
                background:#ecfdf5;
                border:1px solid #a7f3d0;
                color:#065f46;
                font-weight:700;
                margin-top:12px;
                margin-bottom:8px;
                text-align:center;
                line-height:1.7;
                width:100%;
                box-sizing:border-box;
            ">
                분석 완료!<br>
                유사 광고 추출 완료: {n_similar_ads}건
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        analysis_status_placeholder.empty()


        # -------------------------------------------------
    # 4) 저장된 분석 결과 렌더링
    # -------------------------------------------------
    if st.session_state.rec_analysis_done:
        similar_ads_df = st.session_state.rec_similar_ads_df
        final_similar_df = st.session_state.rec_final_similar_df
        optimal_mda_df = st.session_state.rec_optimal_mda_df
        fallback_top_df = st.session_state.rec_fallback_top_df
        result_status = st.session_state.rec_result_status
        meta = st.session_state.rec_meta


        similar_list_cols = get_existing_cols(final_similar_df, SIMILAR_LIST_COLS)
        best_mda_cols = get_existing_cols(optimal_mda_df, BEST_MDA_COLS)
        fallback_show_cols = get_existing_cols(fallback_top_df, BEST_MDA_COLS)

        sort_col = "conversions"

        # -------------------------------------------------
        # 추천 후보 풀 전체(필터 적용 전) 생성
        # -------------------------------------------------
        expand_pool_df = optimal_mda_df[
            optimal_mda_df["action_type_normalized"] == "확대"
        ].copy() if not optimal_mda_df.empty and "action_type_normalized" in optimal_mda_df.columns else pd.DataFrame()

        keep_pool_df = optimal_mda_df[
            optimal_mda_df["action_type_normalized"] == "유지"
        ].copy() if not optimal_mda_df.empty and "action_type_normalized" in optimal_mda_df.columns else pd.DataFrame()

        if not expand_pool_df.empty:
            expand_pool_df = (
                expand_pool_df.sort_values(by=sort_col, ascending=False)
                .drop_duplicates(subset=[MDA_IDX_COL])
                .copy()
            )

        if not keep_pool_df.empty:
            keep_pool_df = (
                keep_pool_df.sort_values(by=sort_col, ascending=False)
                .drop_duplicates(subset=[MDA_IDX_COL])
                .copy()
            )

        

                # -------------------------------------------------
        # 결과 공통 필터용 기준 데이터 (선택 도메인 / 광고유형 기준)
        # -------------------------------------------------
        filter_domain_label = meta.get("selected_domain_label", None)
        filter_ads_type_value = meta.get("selected_ads_type_value", None)

        result_filter_base_df = pd.DataFrame()
        if filter_domain_label is not None and filter_ads_type_value is not None:
            try:
                filter_domain_file = DOMAIN_FILE_MAP[int(filter_domain_label)]
                filter_domain_df = load_csv(filter_domain_file)
                result_filter_base_df = filter_domain_df[
                    pd.to_numeric(filter_domain_df[ADS_TYPE_COL], errors="coerce") == int(filter_ads_type_value)
                ].copy()
            except Exception:
                result_filter_base_df = pd.DataFrame()

        rpc_min, rpc_max = get_slider_min_max(
            result_filter_base_df, RPC_COL, fallback_min=0.0, fallback_max=1.0
        )
        margin_min, margin_max = get_slider_min_max(
            result_filter_base_df, MARGIN_COL, fallback_min=0.0, fallback_max=1.0
        )

        with st.sidebar:
            st.markdown("## 목차")
            st.markdown('<a class="toc-link" href="#search-section">검색하기</a>', unsafe_allow_html=True)
            st.markdown('<a class="toc-link" href="#insight-section">분석 인사이트 보기</a>', unsafe_allow_html=True)
            st.markdown('<a class="toc-link" href="#recommend-section">추천 매체 보기</a>', unsafe_allow_html=True)
            st.markdown('<a class="toc-link" href="#similar-section">유사 광고 정보 보기</a>', unsafe_allow_html=True)

            
        # 검색 조건과 단가 필터 사이 간격
        st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

        # 화면 분할선
        st.markdown(
            """
            <hr style="
                border: none;
                border-top: 1.5px solid #d1d5db;
                margin-top: 0px;
                margin-bottom: 28px;
            ">
            """,
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # 본문 상단 단가 필터
        # -------------------------------------------------
        # -------------------------------------------------
        # 본문 상단 단가 필터
        # -------------------------------------------------
        with st.container(key="filter_panel"):
            title_col, notice_col = st.columns([0.28, 0.72])

            with title_col:
                st.markdown(
                    '<div class="filter-panel-title">단가 필터</div>',
                    unsafe_allow_html=True
                )

            with notice_col:
                st.markdown(
                    """
                    <div style="
                        font-size: 14px;
                        font-weight: 600;
                        color: #b45309;
                        margin-top: 8px;
                        text-align: right;
                    ">
                        ⚠️ 단가 조건을 조정하시면, 아래 분석 인사이트에 반영됩니다
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            filter_col_1, gap_col, filter_col_2 = st.columns([1, 0.5, 1])

            with filter_col_1:
                st.markdown("**클릭당 광고주 단가 범위**")
                st.caption(f"{rpc_min:.1f}원 ~ {rpc_max:.1f}원")

                selected_rpc_range = st.slider(
                    label="클릭당 광고주 단가 범위",
                    min_value=float(rpc_min),
                    max_value=float(rpc_max),
                    value=(float(rpc_min), float(rpc_max)),
                    step=0.1 if (rpc_max - rpc_min) >= 1 else 0.0001,
                    format="%.1f",
                    label_visibility="collapsed",
                    key="main_rpc_range"
                )

                st.caption(
                    f"선택 범위: {selected_rpc_range[0]:.1f}원 ~ {selected_rpc_range[1]:.1f}원"
                )

            with gap_col:
                st.empty()

            with filter_col_2:
                st.markdown("**클릭당 아이브 마진율 범위**")
                st.caption(f"{margin_min * 100:.1f}% ~ {margin_max * 100:.1f}%")

                selected_margin_range = st.slider(
                    label="클릭당 아이브 마진율 범위",
                    min_value=float(margin_min),
                    max_value=float(margin_max),
                    value=(float(margin_min), float(margin_max)),
                    step=0.001 if (margin_max - margin_min) >= 0.01 else 0.0001,
                    format="%.4f",
                    label_visibility="collapsed",
                    key="main_margin_range"
                )

                st.caption(
                    f"선택 범위: {selected_margin_range[0] * 100:.1f}% ~ {selected_margin_range[1] * 100:.1f}%"
                )

        


        

        # -------------------------------------------------
        # 7) 결과 공통 범위 필터 적용
        # -------------------------------------------------
        filtered_expand_pool_df = apply_numeric_range_filter(
            expand_pool_df, RPC_COL, selected_rpc_range[0], selected_rpc_range[1]
        )
        filtered_expand_pool_df = apply_numeric_range_filter(
            filtered_expand_pool_df, MARGIN_COL, selected_margin_range[0], selected_margin_range[1]
        )

        filtered_keep_pool_df = apply_numeric_range_filter(
            keep_pool_df, RPC_COL, selected_rpc_range[0], selected_rpc_range[1]
        )
        filtered_keep_pool_df = apply_numeric_range_filter(
            filtered_keep_pool_df, MARGIN_COL, selected_margin_range[0], selected_margin_range[1]
        )

        fallback_top_df = apply_numeric_range_filter(
            fallback_top_df, RPC_COL, selected_rpc_range[0], selected_rpc_range[1]
        )
        fallback_top_df = apply_numeric_range_filter(
            fallback_top_df, MARGIN_COL, selected_margin_range[0], selected_margin_range[1]
        )

        final_similar_df = apply_numeric_range_filter(
            final_similar_df, RPC_COL, selected_rpc_range[0], selected_rpc_range[1]
        )
        final_similar_df = apply_numeric_range_filter(
            final_similar_df, MARGIN_COL, selected_margin_range[0], selected_margin_range[1]
        )

        # -------------------------------------------------
        # 필터 적용 후 실제 표시용 top 10 생성
        # -------------------------------------------------
        expand_df = filtered_expand_pool_df.head(10).copy()
        keep_df = filtered_keep_pool_df.head(10).copy()

        # -------------------------------------------------
        # 단가 필터 적용 후 표시 상태 재계산
        # -------------------------------------------------
        display_result_status = result_status
      

        if result_status == "expand_exists":
            if not expand_df.empty:
                display_result_status = "expand_exists"
            elif not keep_df.empty:
                display_result_status = "keep_only"
            else:
                display_result_status = "filter_no_match"

        elif result_status == "keep_only":
            if not keep_df.empty:
                display_result_status = "keep_only"
            else:
                display_result_status = "filter_no_match"

        elif result_status in ["no_recommendation", "no_similar_ads"]:
            display_result_status = result_status



        insight_html = build_insight_text(
            result_status=display_result_status,
            query_ad_name=meta.get("query_ad_name", ""),
            selected_domain_name=meta.get("selected_domain_name", ""),
            selected_ads_type_label=meta.get("selected_ads_type_label", ""),
            expand_df=expand_df,
            keep_df=keep_df
        )

    



        render_insight_box(
            insight_html=insight_html,
            icon_base64=samdasu_ai_icon_base64,
            margin_top_px=12,
            margin_bottom_px=100
        )

        st.markdown('<div id="recommend-section"></div>', unsafe_allow_html=True)

        # 1) 우선 추천 매체가 있는 경우
        if display_result_status == "expand_exists":
            st.subheader("우선 추천 매체 top 10")
            expand_display_df = prepare_recommendation_table(expand_df)
            expand_display_df = rename_recommendation_columns(
                expand_display_df[best_mda_cols].reset_index(drop=True)
            )

            st.dataframe(
                expand_display_df,
                use_container_width=True
            )

        elif display_result_status == "keep_only":
            st.subheader("추가 추천 매체 후보 top 10")
            keep_display_df = prepare_recommendation_table(keep_df)
            keep_display_df = rename_recommendation_columns(
                keep_display_df[best_mda_cols].reset_index(drop=True)
            )

            st.dataframe(
                keep_display_df,
                use_container_width=True
            )


        elif display_result_status == "no_recommendation":
            st.subheader("차선책 상위 매체 top 10")
            if fallback_top_df.empty:
                st.warning("차선책으로 보여드릴 상위 매체도 없습니다.")
            else:
                fallback_display_df = prepare_recommendation_table(fallback_top_df)
                fallback_display_df = rename_recommendation_columns(
                    fallback_display_df[fallback_show_cols].reset_index(drop=True)
                )

                st.dataframe(
                    fallback_display_df,
                    use_container_width=True
                )

        elif display_result_status == "no_similar_ads":
            st.subheader("차선책 상위 매체 top 10")
            if fallback_top_df.empty:
                st.warning("차선책으로 보여드릴 상위 매체도 없습니다.")
            else:
                fallback_display_df = prepare_recommendation_table(fallback_top_df)

                st.dataframe(
                    fallback_display_df[fallback_show_cols].reset_index(drop=True),
                    use_container_width=True
                )

        if not final_similar_df.empty:
            st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
            st.markdown('<div id="similar-section"></div>', unsafe_allow_html=True)
            st.subheader("유사 광고 매칭 매체 정보")

            with st.form("similar_match_search_form", clear_on_submit=False):
                search_col_1, search_col_2, search_col_3, search_btn_col, reset_btn_col = st.columns([0.5, 0.5, 0.5, 0.22, 0.2])

                with search_col_1:
                    ads_name_keyword = st.text_input(
                        "광고이름 검색",
                        value=st.session_state.get("rec_match_ads_name_keyword", ""),
                        key="rec_match_ads_name_keyword_input"
                    )

                with search_col_2:
                    ads_idx_keyword = st.text_input(
                        "광고 id 검색",
                        value=st.session_state.get("rec_match_ads_idx_keyword", ""),
                        key="rec_match_ads_idx_keyword_input"
                    )

                with search_col_3:
                    mda_idx_keyword = st.text_input(
                        "매체사 id 검색",
                        value=st.session_state.get("rec_match_mda_idx_keyword", ""),
                        key="rec_match_mda_idx_keyword_input"
                    )

                with search_btn_col:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    search_clicked = st.form_submit_button("🔍 검색", use_container_width=True)

                with reset_btn_col:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    reset_clicked = st.form_submit_button("초기화", use_container_width=True)

            if search_clicked:
                st.session_state["rec_match_ads_name_keyword"] = ads_name_keyword
                st.session_state["rec_match_ads_idx_keyword"] = ads_idx_keyword
                st.session_state["rec_match_mda_idx_keyword"] = mda_idx_keyword

            if reset_clicked:
                st.session_state["rec_match_ads_name_keyword"] = ""
                st.session_state["rec_match_ads_idx_keyword"] = ""
                st.session_state["rec_match_mda_idx_keyword"] = ""

            display_similar_df = final_similar_df.copy()
            display_similar_df = apply_text_contains_filter(
                display_similar_df, "ads_name", st.session_state.get("rec_match_ads_name_keyword", "")
            )
            display_similar_df = apply_text_contains_filter(
                display_similar_df, "ads_idx", st.session_state.get("rec_match_ads_idx_keyword", "")
            )
            display_similar_df = apply_text_contains_filter(
                display_similar_df, "mda_idx", st.session_state.get("rec_match_mda_idx_keyword", "")
            )

            display_similar_df = rename_similar_columns(
                display_similar_df[similar_list_cols].reset_index(drop=True)
            )

            st.dataframe(
                display_similar_df,
                use_container_width=True
            )

        with st.expander("분석 참고 정보"):
            st.write("입력 광고명:", meta.get("query_ad_name", ""))
            st.write("선택 도메인:", meta.get("selected_domain_name", ""), f"({meta.get('selected_domain_label', '')})")
            st.write("선택 광고유형:", meta.get("selected_ads_type_label", ""), f"({meta.get('selected_ads_type_value', '')})")
            st.write("유사 광고 수:", meta.get("n_similar_ads", 0))
            st.write("유사 광고 ads_idx 기준 원본 필터 건수:", meta.get("n_final_rows", 0))
            st.write("최적 운영 매체 건수:", meta.get("n_optimal_rows", 0))
            st.write("결과 상태:", result_status)
            if not similar_ads_df.empty:
                st.write("유사 광고 원본:")
                st.dataframe(similar_ads_df, use_container_width=True)

def render_ranking_page():
    st.title("매체 랭킹")

    # =========================================================
    # 1) 데이터 로드
    # =========================================================
    try:
        rank_df = load_csv(MDA_MASTER_PATH)
    except Exception as e:
        st.error(f"tb_mda_master_ops v2.csv 파일을 불러오지 못했습니다.\n{e}")
        st.stop()

    required_cols = [
        "domain_label", "ads_type", "action_type",
        "mda_idx", "clicks", "conversions", "cvr", "profit"
    ]
    missing_cols = [c for c in required_cols if c not in rank_df.columns]
    if missing_cols:
        st.error(f"tb_mda_master_ops v2.csv 에 필요한 컬럼이 없습니다: {missing_cols}")
        st.stop()

    # =========================================================
    # 2) 상태값 초기화
    # =========================================================
    if "ranking_domain_label" not in st.session_state:
        st.session_state.ranking_domain_label = 1

    if "ranking_ads_type" not in st.session_state:
        domain_init_df = rank_df[rank_df["domain_label"] == st.session_state.ranking_domain_label].copy()
        ads_type_init_vals = (
            pd.to_numeric(domain_init_df["ads_type"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        ads_type_init_vals = sorted(ads_type_init_vals)
        st.session_state.ranking_ads_type = ads_type_init_vals[0] if ads_type_init_vals else None

    # =========================================================
    # 3) 도메인 버튼
    # =========================================================
    st.markdown("### 도메인 선택")

    domain_button_cols = st.columns(5)
    domain_items = [(1, "엔터테인먼트"), (2, "금융"), (3, "라이프스타일"), (4, "커머스"), (5, "기타")]

    for idx, (domain_num, domain_name) in enumerate(domain_items):
        button_label = f"✅ {domain_name}" if st.session_state.ranking_domain_label == domain_num else domain_name
        if domain_button_cols[idx].button(
            button_label,
            key=f"ranking_domain_{domain_num}",
            use_container_width=True
        ):
            st.session_state.ranking_domain_label = domain_num

            # 도메인 바뀌면 해당 도메인의 첫 ads_type으로 초기화
            temp_df = rank_df[rank_df["domain_label"] == domain_num].copy()
            temp_ads_types = (
                pd.to_numeric(temp_df["ads_type"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            temp_ads_types = sorted(temp_ads_types)
            st.session_state.ranking_ads_type = temp_ads_types[0] if temp_ads_types else None
            st.rerun()

    selected_domain_label = st.session_state.ranking_domain_label

    # =========================================================
    # 4) 선택 도메인 데이터
    # =========================================================
    domain_df = rank_df[rank_df["domain_label"] == selected_domain_label].copy()

    if domain_df.empty:
        st.warning("선택한 도메인에 해당하는 데이터가 없습니다.")
        st.stop()

    # =========================================================
    # 5) 광고유형 버튼 (선택 도메인 내 존재하는 값만)
    # =========================================================
    ads_type_values = (
        pd.to_numeric(domain_df["ads_type"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    ads_type_values = sorted(ads_type_values)

    st.markdown("### 광고 유형 선택")

    if not ads_type_values:
        st.warning("선택한 도메인에 광고 유형 데이터가 없습니다.")
        st.stop()

    # 현재 선택 ads_type이 해당 도메인에 없으면 보정
    if st.session_state.ranking_ads_type not in ads_type_values:
        st.session_state.ranking_ads_type = ads_type_values[0]

    ads_type_button_cols = st.columns(min(len(ads_type_values), 6))
    for idx, ads_type_num in enumerate(ads_type_values):
        ads_type_name = ADS_TYPE_LABEL_MAP.get(ads_type_num, f"기타({ads_type_num})")
        button_label = (
            f"✅ {ads_type_name}"
            if st.session_state.ranking_ads_type == ads_type_num
            else ads_type_name
        )

        if ads_type_button_cols[idx % len(ads_type_button_cols)].button(
            button_label,
            key=f"ranking_ads_type_{selected_domain_label}_{ads_type_num}",
            use_container_width=True
        ):
            st.session_state.ranking_ads_type = ads_type_num
            st.rerun()

    selected_ads_type = st.session_state.ranking_ads_type
    selected_ads_type_name = ADS_TYPE_LABEL_MAP.get(selected_ads_type, f"기타({selected_ads_type})")
    selected_domain_name = DOMAIN_LABEL_MAP_REV.get(selected_domain_label, f"도메인 {selected_domain_label}")

    # =========================================================
    # 6) 최종 필터링
    # =========================================================
    filtered_df = domain_df[
        pd.to_numeric(domain_df["ads_type"], errors="coerce") == selected_ads_type
    ].copy()

    filtered_df["action_type_normalized"] = filtered_df["action_type"].map(normalize_action_type)

    expand_df = filtered_df[filtered_df["action_type_normalized"] == "확대"].copy()
    keep_df = filtered_df[filtered_df["action_type_normalized"] == "유지"].copy()

    # TOP 10 기준: profit 내림차순
    expand_top10 = expand_df.sort_values(by="profit", ascending=False).head(10).copy()
    keep_top10 = keep_df.sort_values(by="profit", ascending=False).head(10).copy()

    # =========================================================
    # 7) 인사이트 박스
    # =========================================================
    total_count = len(filtered_df)
    expand_count = len(expand_df)
    keep_count = len(keep_df)
    total_profit = filtered_df["profit"].sum() if "profit" in filtered_df.columns else 0
    expand_profit = expand_df["profit"].sum() if "profit" in expand_df.columns else 0
    keep_profit = keep_df["profit"].sum() if "profit" in keep_df.columns else 0

    st.info(
        f"""
**선택 결과 요약**

- 현재 선택: **{selected_domain_name} / {selected_ads_type_name}**
- 전체 매체 수: **{total_count:,}개**
- 확대 매체 수: **{expand_count:,}개**
- 유지 매체 수: **{keep_count:,}개**
- 전체 profit 합계: **{total_profit:,.0f}**
- 확대 profit 합계: **{expand_profit:,.0f}**
- 유지 profit 합계: **{keep_profit:,.0f}**

현재 화면의 TOP 10은 **profit 기준 내림차순**으로 정렬되어 있습니다.
        """
    )

    # =========================================================
    # 8) 좌우 테이블
    # =========================================================
    show_cols = get_existing_cols(
        filtered_df,
        ["mda_idx", "clicks", "conversions", "cvr", "profit"]
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("고효율 확정 매체 TOP 10")
        if expand_top10.empty:
            st.warning("확대 매체가 없습니다.")
        else:
            st.dataframe(
                expand_top10[show_cols].reset_index(drop=True),
                use_container_width=True
            )

    with right_col:
        st.subheader("고효율 후보 매체(추가 관찰 필요) TOP 10")
        if keep_top10.empty:
            st.warning("유지 매체가 없습니다.")
        else:
            st.dataframe(
                keep_top10[show_cols].reset_index(drop=True),
                use_container_width=True
            )

def render_management_page():
    st.title("관리 필요 매체")
    st.info("이 페이지는 화면 분할만 먼저 구성한 상태입니다. 다음 단계에서 관리 필요 기준과 상세 테이블을 연결하면 됩니다.")
    st.markdown("- 추후 배치 예정: 저효율/개선우선/축소후보 매체 목록, 경고 신호 요약")



# =========================================================
# 15) 사이드바 네비게이션 (박스형 버튼 메뉴)
# =========================================================
st.markdown("""
<style>
section[data-testid="stSidebar"] button {
    border: 1px solid #d9d9d9 !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.6rem !important;
    background-color: white !important;
    color: black !important;
    font-weight: 600 !important;
    text-align: left !important;
    width: 100% !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] button:hover {
    border-color: #999999 !important;
    background-color: #f5f5f5 !important;
}
</style>
""", unsafe_allow_html=True)

render_recommendation_page()

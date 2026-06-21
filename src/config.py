import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 로컬 .env 로드
load_dotenv()

# 경로 설정
if getattr(sys, 'frozen', False):
    # PyInstaller로 패키징된 실행 파일 경로
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    # 일반 파이썬 실행
    BASE_DIR = Path(__file__).resolve().parent.parent

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# 대상 종목 코드
# KODEX 200의 yfinance 티커는 '069500.KS'
TICKER_KODEX200 = "069500.KS"

# 대형 영향력 주식
TICKER_SAMSUNG = "005930.KS"     # 삼성전자
TICKER_HYNIX = "000660.KS"       # SK하이닉스

# 거시경제 수집용 야간선물/해외지수 티커
TICKER_KOSPI_FUTURE = "^KS200"  # 코스피 200 지수
TICKER_NASDAQ_FUTURE = "NQ=F"   # 나스닥 100 선물
TICKER_USD_KRW = "USDKRW=X"      # 원/달러 환율
TICKER_USD_JPY = "JPY=X"         # 달러/엔 환율
TICKER_NIKKEI = "^N225"          # 일본 닛케이 225 지수
TICKER_US10Y = "^TNX"            # 미국 10년물 국채 금리
TICKER_WTI = "CL=F"              # WTI 크루드 오일 선물
TICKER_VIX = "^VIX"              # CBOE Volatility Index (공포 지수)
TICKER_SOX = "^SOX"              # 필라델피아 반도체 지수
TICKER_SHANGHAI = "000001.SS"      # 상해 종합 지수
TICKER_DOLLAR = "DX-Y.NYB"       # US 달러 인덱스
TICKER_GOLD = "GC=F"             # 금 선물
TICKER_US_CPI = "CPIAUCSL"       # 미국 소비자 물가지수 (FRED)
TICKER_EWY = "EWY"               # MSCI 한국 ETF
TICKER_BITCOIN = "BTC-USD"       # 비트코인
TICKER_US_RATE = "FEDFUNDS"      # 미국 기준금리 (FRED)
TICKER_NASDAQ = "^IXIC"          # 나스닥 종합지수

# AI 모델 목록
AI_MODELS = {
    "Gemini": "gemini-1.5-pro",
    "ChatGPT": "gpt-4o",
    "Claude": "claude-3-5-sonnet-20240620",
    "Grok": "grok-2"
}

# 모델별 가중치 설정 (전문성 기반)
# Gemini: 30%, ChatGPT: 30%, Claude: 25%, Grok: 15%
DEFAULT_WEIGHTS = {
    "Gemini": 0.25,
    "ChatGPT": 0.25,
    "Claude": 0.25,
    "Grok": 0.25
}

# .env 파일에서 로드된 API 키 (GUI 미설정 시 폴백용)
ENV_API_KEYS = {
    "Gemini": os.environ.get("GEMINI_API_KEY", ""),
    "ChatGPT": os.environ.get("OPENAI_API_KEY", ""),
    "Claude": os.environ.get("CLAUDE_API_KEY", ""),
    "Grok": os.environ.get("GROK_API_KEY", "")
}

# 매크로 지표 반영 가중치 설정 (가중치 초기값은 모두 0%로 설정)
MACRO_WEIGHTS = {
    "Kospi_Future": 0.00,
    "Nasdaq_Future": 0.00,
    "Kodex200": 0.00,           # Kodex200 (S&P 500 선물 대체)
    "USD_KRW": 0.00,
    "USD_JPY": 0.00,
    "VIX_Index": 0.00,
    "US10Y_Treasury": 0.00,
    "WTI_Crude": 0.00,
    "KR_Rate": 0.00,
    "KR_Bond": 0.00,
    "SOX_Index": 0.00,
    "Dollar_Index": 0.00,
    "Gold_Future": 0.00,
    "US_CPI": 0.00,
    "Technical_Analysis1": 0.00,
    "Technical_Analysis2": 0.00,
    "Nikkei_225": 0.00,
    "Shanghai_Composite": 0.00,
    "MSCI_Korea": 0.00,
    "Short_Selling": 0.00,
    "Famous_Remarks": 0.00,
    "Bitcoin": 0.00,
    "US_Rate": 0.00,
    "NASDAQ": 0.00,
}

# 기본 매크로 상대적 가중치 기준 (동적 연산용)
BASE_MACRO_WEIGHTS = {
    "Kospi_Future": 0.28,
    "Nasdaq_Future": 0.15,
    "Kodex200": 0.08,
    "USD_KRW": -0.04,
    "USD_JPY": -0.03,
    "VIX_Index": -0.02,
    "US10Y_Treasury": -0.01,
    "WTI_Crude": -0.01,
    "KR_Rate": -0.05,
    "KR_Bond": -0.04,
    "SOX_Index": 0.05,
    "Dollar_Index": -0.03,
    "Gold_Future": 0.02,
    "US_CPI": -0.01,
    "Technical_Analysis1": 0.05,
    "Technical_Analysis2": 0.05,
    "Nikkei_225": 0.01,
    "Shanghai_Composite": 0.01,
    "MSCI_Korea": 0.02,
    "Short_Selling": -0.02,
    "Famous_Remarks": 0.01,
    "Bitcoin": 0.01,
    "US_Rate": 0.00,
    "NASDAQ": 0.00,
}

# 매크로 지표 한글 레이블 매핑
MACRO_LABELS = {
    "Kospi_Future": "코스피 선물",
    "Nasdaq_Future": "나스닥 선물",
    "Kodex200": "Kodex200",
    "USD_KRW": "원/달러 환율",
    "USD_JPY": "엔/달러 환율",
    "VIX_Index": "VIX 공포지수",
    "US10Y_Treasury": "미 10년 국채금리",
    "WTI_Crude": "WTI 국제 유가",
    "KR_Rate": "국내 금리",
    "KR_Bond": "국내 채권금리",
    "SOX_Index": "필라델피아 반도체",
    "Dollar_Index": "달러 인덱스",
    "Gold_Future": "금 선물",
    "US_CPI": "미국소비자물가지수",
    "Technical_Analysis1": "기술적분석1",
    "Technical_Analysis2": "기술적분석2",
    "Nikkei_225": "일본 닛케이",
    "Shanghai_Composite": "상해 종합지수",
    "MSCI_Korea": "MSCI 한국 ETF 종가",
    "Short_Selling": "상위종목공매도",
    "Famous_Remarks": "유명인사 발언",
    "Bitcoin": "비트코인",
    "US_Rate": "미국 금리",
    "NASDAQ": "나스닥"
}


# 최종 예측 결론 반영 비율 가중치 (총합 = 1.0)
CONSENSUS_WEIGHTS = {
    "AI_Consensus": 0.50,       # 4대 핵심 AI 예측 분석
    "Macro_Dashboard": 0.30,   # 글로벌 매크로 실시간 대시보드
    "News_Consensus": 0.15,     # 실시간 속보 뉴스
    "Rumor_Consensus": 0.05     # 증권가 소문/이슈
}


# KST 시간 도우미 함수 (Huggingface 등 해외 서버 환경에서 한국 시간으로 일관되게 표시하기 위함)
def get_kst_now():
    """한국 시간(KST, UTC+9)의 현재 datetime을 timezone-naive 객체로 반환합니다.
    다른 naive datetime 객체들과의 비교 및 호환을 위해 tzinfo를 제거합니다.
    """
    from datetime import datetime, timezone, timedelta
    kst_tz = timezone(timedelta(hours=9))
    return datetime.now(kst_tz).replace(tzinfo=None)

def get_kst_today():
    """한국 시간(KST) 기준의 오늘 날짜를 date 객체로 반환합니다."""
    return get_kst_now().date()




import os
from pathlib import Path
from dotenv import load_dotenv

# 로컬 .env 로드
load_dotenv()

# 경로 설정
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
TICKER_SP500_FUTURE = "ES=F"    # S&P 500 선물
TICKER_USD_KRW = "USDKRW=X"      # 원/달러 환율
TICKER_USD_JPY = "JPY=X"         # 달러/엔 환율
TICKER_NIKKEI = "^N225"          # 일본 닛케이 225 지수
TICKER_US10Y = "^TNX"            # 미국 10년물 국채 금리
TICKER_WTI = "CL=F"              # WTI 크루드 오일 선물
TICKER_VIX = "^VIX"              # CBOE Volatility Index (공포 지수)

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

# 매크로 지표 및 AI 컨센서스 반영 가중치 설정 (가중치 절대값의 합이 1.0 내외로 정규화됨)
# 선물 지수가 가장 영향력이 크도록 설정 (코스피선물 30%, 나스닥선물 20%, S&P선물 10% = 선물 합계 60%)
MACRO_WEIGHTS = {
    "AI_Consensus": 0.20,       # 4대 AI 종합 의견
    "Kospi_Future": 0.30,       # 코스피 200 선물 (가장 높은 가중치)
    "Nasdaq_Future": 0.20,      # 나스닥 100 선물
    "SP500_Future": 0.10,       # S&P 500 선물
    "USD_KRW": -0.05,           # 원/달러 환율 (상승 시 악재이므로 음수 가중치)
    "USD_JPY": -0.03,           # 달러/엔 환율 (상승 시 수출 경합도 부정적이므로 엔화 약세 악재)
    "Nikkei_225": 0.04,         # 일본 닛케이 지수 (동조화)
    "VIX_Index": -0.05,          # VIX 공포지수 (상승 시 심리 위축 악재이므로 음수 가중치)
    "US10Y_Treasury": -0.02,    # 미 10년 국채금리 (상승 시 악재이므로 음수 가중치)
    "WTI_Crude": -0.01,         # WTI 유가 (상승 시 비용 증가 악재이므로 음수 가중치)
}



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


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
TICKER_SP500_FUTURE = "ES=F"    # S&P 500 선물
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
TICKER_EUR_USD = "EURUSD=X"      # 유로/달러 환율

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

# 매크로 지표 반영 가중치 설정 (가중치 절대값의 합이 1.0 = 100%가 되도록 조정)
MACRO_WEIGHTS = {
    "Kospi_Future": 0.28,       # 코스피 200 선물 (기존 0.30 -> 0.28)
    "Nasdaq_Future": 0.15,      # 나스닥 100 선물 (기존 0.18 -> 0.15)
    "SP500_Future": 0.08,       # S&P 500 선물 (기존 0.10 -> 0.08)
    "USD_KRW": -0.04,           # 원/달러 환율 (기존 -0.05 -> -0.04)
    "USD_JPY": -0.03,           # 달러/엔 환율
    "VIX_Index": -0.02,          # VIX 공포지수
    "US10Y_Treasury": -0.01,    # 미 10년 국채금리
    "WTI_Crude": -0.01,         # WTI 유가
    "KR_Rate": -0.05,           # 국내 금리 (한국은행 기준금리)
    "KR_Bond": -0.04,           # 국내 채권금리 (국고채 3년 금리)
    "SOX_Index": 0.05,          # 필라델피아 반도체 지수
    "Dollar_Index": -0.03,      # 달러 인덱스
    "Gold_Future": 0.02,        # 금 선물
    "US_CPI": -0.01,            # 미국 소비자 물가지수
    "Technical_Analysis1": 0.05,  # 기술적 분석1 (Kodex200 최근주가분석)
    "Technical_Analysis2": 0.05,  # 기술적 분석2 (Kodex200 MA,MACD,RSI)
    "Nikkei_225": 0.01,         # 일본 닛케이 지수
    "Shanghai_Composite": 0.01,  # 상해 종합 지수
    "MSCI_Korea": 0.02,         # MSCI 한국 ETF 종가 (기존 대만 가권 대체)
    "Short_Selling": -0.02,     # 상위종목공매도 (기존 홍콩 항셍 대체)
    "Famous_Remarks": 0.01,     # 유명인사 발언 (기존 유로스톡스 50 대체)
    "Bitcoin": 0.01,            # 비트코인
    "US_Rate": 0.00,            # 미국 금리 (검토중)
    "EUR_USD": 0.00,            # 검토중
}

# 매크로 지표 한글 레이블 매핑
MACRO_LABELS = {
    "Kospi_Future": "코스피 선물",
    "Nasdaq_Future": "나스닥 선물",
    "SP500_Future": "S&P 500 선물",
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
    "EUR_USD": "검토중"
}


# 최종 예측 결론 반영 비율 가중치 (총합 = 1.0)
CONSENSUS_WEIGHTS = {
    "AI_Consensus": 0.50,       # 4대 핵심 AI 예측 분석
    "Macro_Dashboard": 0.30,   # 글로벌 매크로 실시간 대시보드
    "News_Consensus": 0.15,     # 실시간 속보 뉴스
    "Rumor_Consensus": 0.05     # 증권가 소문/이슈
}



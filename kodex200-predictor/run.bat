@echo off
chcp 65001 > nul
echo ===================================================
echo   KODEX 200 AI Stock Predictor 가동 스크립트
echo ===================================================
echo.
echo [1/2] 필요한 패키지를 설치/업데이트 중입니다...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [경고] 패키지 설치 과정에서 에러가 발생했으나 계속 진행합니다.
)
echo.
echo [2/2] KODEX 200 AI 예측 프로그램을 시작합니다...
python main.py
if %errorlevel% neq 0 (
    echo.
    echo [오류] 프로그램 실행에 실패했습니다. 파이썬 환경 또는 Flet 설치를 확인하세요.
)
echo.
pause

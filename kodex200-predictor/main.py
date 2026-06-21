import os
import sys
import warnings

# 현재 경로를 sys.path에 추가하여 패키지 가져오기가 원활하도록 처리
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Flet 0.85 deprecation 경고 숨김
warnings.filterwarnings("ignore", category=DeprecationWarning)

import flet as ft
from src.app import main

if __name__ == "__main__":
    # 포트 설정 (Hugging Face Spaces는 기본적으로 7860 포트를 요구하며 SPACE_ID 환경변수가 존재함)
    port = int(os.environ.get("PORT", 8550))
    is_cloud = bool(os.environ.get("PORT") or os.environ.get("SPACE_ID"))
    if os.environ.get("SPACE_ID"):
        port = 7860
        
    ft.app(
        target=main,
        port=port,
        view=ft.AppView.WEB_BROWSER if is_cloud else None
    )

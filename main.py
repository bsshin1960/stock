import sys
import os
import warnings

# 현재 경로를 sys.path에 추가하여 패키지 가져오기가 원활하도록 처리
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Flet 0.85 deprecation 경고 숨김
warnings.filterwarnings("ignore", category=DeprecationWarning)

import flet as ft
from src.app import main

if __name__ == "__main__":
    ft.app(target=main)

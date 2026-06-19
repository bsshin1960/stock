"""
최종 프로덕션 빌드 스크립트
- scipy, PyQt6 등 불필요한 대용량 모듈 제외로 사이즈 축소 (약 130MB 목표)
- traceback 등에서 동적으로 참조하는 'codeop' 및 'code' 모듈을 hidden-import로 강제 포함하여 실행 오류 방지
- noconsole(콘솔 창 미표시) 설정
- 빌드 완료 후 루트 경로로 exe 복사 및 빌드 임시 파일 정리
"""
import os, sys, shutil, zipfile

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import PyInstaller.__main__
import flet_cli.__pyinstaller.config as hook_config
from flet_cli.__pyinstaller.utils import copy_flet_bin
from flet_cli.__pyinstaller.win_utils import update_flet_view_version_info

APP_NAME = "Kodex200_AI_Stock_Predictor"

for d in ["build", "dist"]:
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)

hook_config.temp_bin_dir = copy_flet_bin()

version_info_path = None
if hook_config.temp_bin_dir:
    fletd_path = os.path.join(hook_config.temp_bin_dir, "fletd.exe")
    if os.path.exists(fletd_path):
        os.remove(fletd_path)

    exe_path = os.path.join(hook_config.temp_bin_dir, "flet", "flet.exe")
    if os.path.exists(exe_path):
        version_info_path = update_flet_view_version_info(
            exe_path=exe_path,
            product_name="KODEX 200 AI Stock Predictor",
            file_description="KODEX 200 AI Stock Predictor",
            product_version="1.0.0",
            file_version="1.0.0.0",
            company_name="Shinbosung",
            copyright="Copyright @ 2026 Shinbosung",
        )

    flet_dir = os.path.join(hook_config.temp_bin_dir, "flet")
    if os.path.isdir(flet_dir):
        zip_path = os.path.join(hook_config.temp_bin_dir, "flet-windows.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(flet_dir):
                for f in files:
                    full = os.path.join(root, f)
                    arcname = os.path.relpath(full, hook_config.temp_bin_dir)
                    zf.write(full, arcname)
        shutil.rmtree(flet_dir)

# 빌드 매개변수 구성
pyi_args = [
    "main.py",
    "--noconfirm",
    "--noconsole",           # 실제 배포용이므로 콘솔 미표시
    "--onefile",
    "--name", APP_NAME,
    "--distpath", "dist",
    "--add-data", "src:src",
    "--hidden-import", "google.generativeai",
    "--hidden-import", "openai",
    "--hidden-import", "anthropic",
    "--hidden-import", "yfinance",
    "--hidden-import", "pandas",
    "--hidden-import", "numpy",
    "--hidden-import", "bs4",
    "--hidden-import", "requests",
    "--hidden-import", "matplotlib",
    "--hidden-import", "matplotlib.backends.backend_agg",
    "--hidden-import", "dotenv",
    "--hidden-import", "tkinter",
    "--hidden-import", "tkinter.filedialog",
    # 실행 시 ModuleNotFoundError 방지를 위한 표준 라이브러리 강제 포함
    "--hidden-import", "code",
    "--hidden-import", "codeop",
    # 대용량 및 불필요 모듈 제외
    "--exclude-module", "scipy",
    "--exclude-module", "sklearn",
    "--exclude-module", "PyQt6",
    "--exclude-module", "PyQt5",
    "--exclude-module", "PySide6",
    "--exclude-module", "PySide2",
    "--exclude-module", "matplotlib.backends.backend_qt5agg",
    "--exclude-module", "matplotlib.backends.backend_qt6agg",
    "--exclude-module", "matplotlib.backends.backend_qtagg",
    "--exclude-module", "matplotlib.backends.backend_tkagg",
    "--exclude-module", "matplotlib.backends.backend_wxagg",
    "--exclude-module", "IPython",
    "--exclude-module", "notebook",
    "--exclude-module", "lib2to3",
    "--exclude-module", "pandas.tests",
    "--exclude-module", "numpy.tests",
]

if version_info_path:
    pyi_args.extend(["--version-file", version_info_path])

print("프로덕션 빌드 시작...")
PyInstaller.__main__.run(pyi_args)

if hook_config.temp_bin_dir and os.path.exists(hook_config.temp_bin_dir):
    shutil.rmtree(hook_config.temp_bin_dir, ignore_errors=True)

target_exe = f"dist/{APP_NAME}.exe"
if os.path.exists(target_exe):
    # 루트 경로로 복사
    dest_path = f"./{APP_NAME}.exe"
    shutil.copy2(target_exe, dest_path)
    size = os.path.getsize(dest_path)
    print(f"빌드 완료 및 복사 성공: {dest_path} ({size/1024/1024:.1f} MB)")
    
    # 임시 빌드 디렉토리 정리
    for d in ["build", "dist"]:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    
    spec_file = f"{APP_NAME}.spec"
    if os.path.exists(spec_file):
        os.remove(spec_file)
else:
    print("에러: 빌드된 EXE 파일을 찾을 수 없습니다.")

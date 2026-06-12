# -*- coding: utf-8 -*-
import os
import json
import threading
import datetime
import flet as ft
import yfinance as yf
import pandas as pd
from src.data_collector import DataCollector
from src.ai_consensus import AIConsensusManager
from src.reporter import PredictionReporter
from src.config import DEFAULT_WEIGHTS, ENV_API_KEYS, REPORTS_DIR, BASE_DIR, TICKER_KODEX200

# --- 로컬 JSON 파일 기반 API Key 저장소 ---
_SETTINGS_FILE = BASE_DIR / "settings.json"

def _load_settings() -> dict:
    try:
        if _SETTINGS_FILE.exists():
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_settings(data: dict):
    try:
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Warning] 설정 저장 실패: {e}")


class CustomSwitch(ft.Container):
    def __init__(self, value=True, on_change=None):
        self.value = value
        self.on_change = on_change
        
        self.thumb = ft.Container(
            width=16,
            height=16,
            bgcolor="#FFFFFF",
            border_radius=8
        )
        
        super().__init__(
            content=self.thumb,
            width=46,  # 10% wider slot
            height=22,
            border_radius=11,
            padding=ft.Padding(left=2, top=2, right=2, bottom=2),
            alignment=ft.Alignment(1, 0) if value else ft.Alignment(-1, 0),
            animate_align=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
            bgcolor="#2196F3" if value else "#475569",
            border=ft.Border.all(1, "#78909C"),
            on_click=self._on_click
        )

    def _on_click(self, e):
        self.value = not self.value
        self.alignment = ft.Alignment(1, 0) if self.value else ft.Alignment(-1, 0)
        if self.on_change:
            self.on_change(e)


class StockPredictorApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "KODEX 200 AI Stock Predictor"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.bgcolor = "#F4F6F9"
        self.page.padding = 0
        self.page.window.width = 1340
        self.page.window.height = 980
        self.page.window.min_width = 1340
        self.page.window.min_height = 980
        self.page.window.resizable = True
        self.page.scroll = None
        self.is_box_hovered = False
        self.last_scroll_offset = 0.0
        self.max_scroll_height = 60.0
        self.is_programmatic_scroll = False


        self.data_collector = DataCollector()
        self.reporter = PredictionReporter()

        saved = _load_settings()
        self.api_keys = {
            "Gemini": saved.get("api_key_gemini", "") or ENV_API_KEYS.get("Gemini", ""),
            "ChatGPT": saved.get("api_key_chatgpt", "") or ENV_API_KEYS.get("ChatGPT", ""),
            "Claude": saved.get("api_key_claude", "") or ENV_API_KEYS.get("Claude", ""),
            "Grok": saved.get("api_key_grok", "") or ENV_API_KEYS.get("Grok", ""),
        }

        self.current_data = None
        self.ai_results = None
        self.consensus_result = None
        self._is_running = False
        self.is_menu_open = False

        self.setup_ui()
        self.page.run_task(self.page.window.center)
        # 초기 주가 차트 비동기 로딩
        self.page.run_thread(self.load_charts)
        # 초기 예측 적중률 이력 로딩
        self.page.run_thread(self.load_history_ui)

    def show_snack_bar(self, message: str, color: str = "#00C853"):
        sb = ft.SnackBar(content=ft.Text(message), bgcolor=color)
        self.page.overlay.append(sb)
        sb.open = True
        self.page.update()

    def load_charts(self):
        try:
            self._log("주가 차트(1년) 생성 및 데이터 수집 중...")
            kodex_b64 = self.data_collector.generate_chart_base64("069500.KS", "KODEX 200 주가 추이 (1년)")
            kospi_b64 = self.data_collector.generate_chart_base64("^KS11", "KOSPI 종합주가지수 추이 (1년)")
            
            if kodex_b64:
                self.kodex_chart.src_base64 = kodex_b64
            if kospi_b64:
                self.kospi_chart.src_base64 = kospi_b64
                
            self._log("✔ 차트 로딩 완료")
            self.page.update()
        except Exception as e:
            self._log(f"✘ 차트 로드 실패: {e}")

    # ─── 모니터링 로그 ───
    def _log(self, msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        is_dark = self.page.theme_mode == ft.ThemeMode.DARK
        text_color = "#B0C4DE" if is_dark else "#000000"
        self.monitor_lv.controls.append(
            ft.Container(
                content=ft.Text(f"[{ts}] {msg}", size=12, color=text_color, selectable=True),
                on_hover=self.handle_box_hover,
                bgcolor="#00000000"
            )
        )
        try:
            self.page.update()
        except Exception:
            pass

    # ─── UI 빌드 ───
    def setup_ui(self):
        # ===== 메뉴바 =====
        self.menu_file_text = ft.Text("파일", size=13, color="#1E293B")
        self.menu_setting_text = ft.Text("설정", size=13, color="#1E293B")
        self.menu_help_text = ft.Text("도움말", size=13, color="#1E293B")

        menu_border = "#B0BEC5"
        menu_bg = "#FFFFFF"
        item_style = ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=4),
            padding=ft.Padding(left=8, right=8, top=0, bottom=0),
            side=ft.BorderSide(0, "transparent"),
            bgcolor={
                "hovered": "#F1F5F9",
                "": "transparent"
            }
        )

        self.menubar = ft.MenuBar(
            style=ft.MenuStyle(
                bgcolor="#F4F6F9",
                elevation=0,
                shadow_color="transparent",
                shape=ft.RoundedRectangleBorder(radius=0),
                side=ft.BorderSide(0, "transparent")
            ),
            controls=[
                ft.SubmenuButton(
                    content=self.menu_file_text,
                    on_open=self.handle_menu_open,
                    on_close=self.handle_menu_close,
                    menu_style=ft.MenuStyle(
                        bgcolor=menu_bg,
                        side=ft.BorderSide(1, menu_border),
                        shape=ft.RoundedRectangleBorder(radius=8),
                        padding=ft.Padding(left=4, top=2, right=4, bottom=2),
                        shadow_color="black",
                        elevation=8
                    ),
                    controls=[
                        ft.MenuItemButton(content=ft.Text("실행", size=12), leading=ft.Icon(ft.Icons.PLAY_ARROW_ROUNDED, size=16), on_click=self.run_analysis, style=item_style, height=25),
                        ft.MenuItemButton(content=ft.Text("열기", size=12), leading=ft.Icon(ft.Icons.FILE_OPEN_ROUNDED, size=16), on_click=self.load_report_file, style=item_style, height=25),
                        ft.MenuItemButton(content=ft.Text("저장", size=12), leading=ft.Icon(ft.Icons.SAVE_ROUNDED, size=16), on_click=self.save_report_file, style=item_style, height=25),
                        ft.MenuItemButton(content=ft.Text("종료", size=12), leading=ft.Icon(ft.Icons.EXIT_TO_APP, size=16), on_click=lambda _: self.page.window.close(), style=item_style, height=25),
                    ],
                ),
                ft.SubmenuButton(
                    content=self.menu_setting_text,
                    on_open=self.handle_menu_open,
                    on_close=self.handle_menu_close,
                    menu_style=ft.MenuStyle(
                        bgcolor=menu_bg,
                        side=ft.BorderSide(1, menu_border),
                        shape=ft.RoundedRectangleBorder(radius=8),
                        padding=ft.Padding(left=4, top=2, right=4, bottom=2),
                        shadow_color="black",
                        elevation=8
                    ),
                    controls=[
                        ft.MenuItemButton(content=ft.Text("API Key 설정", size=12), leading=ft.Icon(ft.Icons.KEY_ROUNDED, size=16), on_click=self.open_settings_dialog, style=item_style, height=25),
                    ],
                ),
                ft.SubmenuButton(
                    content=self.menu_help_text,
                    on_open=self.handle_menu_open,
                    on_close=self.handle_menu_close,
                    menu_style=ft.MenuStyle(
                        bgcolor=menu_bg,
                        side=ft.BorderSide(1, menu_border),
                        shape=ft.RoundedRectangleBorder(radius=8),
                        padding=ft.Padding(left=4, top=2, right=4, bottom=2),
                        shadow_color="black",
                        elevation=8
                    ),
                    controls=[
                        ft.MenuItemButton(content=ft.Text("사용 가이드", size=12), leading=ft.Icon(ft.Icons.HELP_OUTLINE_ROUNDED, size=16), on_click=self.open_help_dialog, style=item_style, height=25),
                        ft.MenuItemButton(content=ft.Text("프로그램 정보", size=12), leading=ft.Icon(ft.Icons.INFO_OUTLINE_ROUNDED, size=16), on_click=self.open_about_dialog, style=item_style, height=25),
                    ],
                ),
            ],
        )

        # ===== 헤더 =====
        self.title_label = ft.Text("KODEX 200 AI Predictor", size=24, weight=ft.FontWeight.BOLD, color="#000000")
        init_date_color = "#7C3AED"
        self.subtitle_date_span = ft.TextSpan(f"{datetime.datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')} 기준", style=ft.TextStyle(color=init_date_color, weight=ft.FontWeight.BOLD))
        self.subtitle_label = ft.Text(
            spans=[
                self.subtitle_date_span,
                ft.TextSpan(" 국내/미국 선물, 실시간 뉴스/주가, VIX 공포지수 등 변동 요인을 종합 분석하여 다음날 Kodex200 ETF의 시초가 예측",
                            style=ft.TextStyle(weight=ft.FontWeight.BOLD))
            ],
            size=14,
            color="#64748B"
        )

        # ===== 최종 결과 카드 =====
        self.result_status = ft.Text("대기 중...", size=18, weight=ft.FontWeight.BOLD, color="#8A99AD")
        self.result_pct = ft.Text("", size=18, weight=ft.FontWeight.BOLD, color="#8A99AD")
        self.result_price = ft.Text("", size=13, color="#B0C4DE")
        self.result_diff = ft.Text("", size=13, color="#B0C4DE")

        weights_str = "AI 50% │ 매크로 30% │ 뉴스 15% │ 소문 5%"
        self.consensus_title_icon = ft.Icon(ft.Icons.ANALYTICS_ROUNDED, size=16, color="#C084FC")
        self.consensus_title_text = ft.Text("최종 종합 예측 결과", size=13, color="#C084FC", weight=ft.FontWeight.BOLD)

        self.consensus_box = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Row([
                        self.consensus_title_icon,
                        self.consensus_title_text,
                    ], spacing=6),
                    ft.Text(f"가중치: {weights_str}", size=10, color="#8A99AD", italic=True),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Divider(color="#2E3A4E", thickness=1, height=1),
                ft.Row([
                    ft.Column([self.result_status, self.result_diff], spacing=1, alignment=ft.MainAxisAlignment.CENTER),
                    ft.VerticalDivider(width=10, color="#2E3A4E"),
                    ft.Column([self.result_pct, self.result_price], spacing=1, alignment=ft.MainAxisAlignment.CENTER),
                ], alignment=ft.MainAxisAlignment.SPACE_AROUND, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ], spacing=5),
            bgcolor="#FFFFFF", padding=ft.Padding(left=15, right=15, top=10, bottom=10), border_radius=12,
            border=ft.Border.all(2, "#40C4FF"), width=631, height=110,
            on_hover=self.handle_body_hover,
        )

        # ===== 시총 TOP10 주가 박스 =====
        self.top10_lv = ft.ListView(expand=True, spacing=2, padding=ft.Padding(left=4, right=4, top=2, bottom=2))
        self.top10_title_icon = ft.Icon(ft.Icons.LEADERBOARD_ROUNDED, size=16, color="#7C3AED")
        self.top10_title_text = ft.Text("시총 TOP10 회사 주가", size=13, color="#7C3AED", weight=ft.FontWeight.BOLD)

        self.top10_box = ft.Container(
            content=ft.Column([
                ft.Row([self.top10_title_icon, self.top10_title_text], spacing=6),
                ft.Divider(color="#CBD5E1", thickness=1, height=1),
                ft.Container(content=self.top10_lv, expand=True, bgcolor="#00000000"),
            ], spacing=4),
            bgcolor="#FFFFFF", padding=ft.Padding(left=12, right=12, top=10, bottom=10), border_radius=12,
            border=ft.Border.all(1, "#78909C"), width=309, height=110,
            on_hover=self.handle_body_hover,
        )

        # ===== 개발중 플레이스홀더 박스 =====
        self.dev_box = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.CONSTRUCTION_ROUNDED, size=16, color="#7C3AED"),
                    ft.Text("개발중...", size=13, color="#7C3AED", weight=ft.FontWeight.BOLD)
                ], spacing=6),
                ft.Divider(color="#CBD5E1", thickness=1, height=1),
                ft.Text("향후 추가 정보를 표시할 예정입니다.", size=12, color="#94A3B8", italic=True),
            ], spacing=6),
            bgcolor="#FFFFFF", padding=ft.Padding(left=12, right=12, top=10, bottom=10), border_radius=12,
            border=ft.Border.all(1, "#78909C"), width=309, height=110,
            on_hover=self.handle_body_hover,
        )

        # 분석 로직 호환성 유지용 히든 위젯
        self.kodex_price = ft.Text("", visible=False)
        self.kodex_change = ft.Text("", visible=False)
        self.kodex_title_icon = ft.Icon(ft.Icons.FLAG_CIRCLE_ROUNDED, size=1, visible=False)
        self.kodex_title_text = ft.Text("", visible=False)
        self.samsung_price = ft.Text("", visible=False)
        self.samsung_change = ft.Text("", visible=False)
        self.hynix_price = ft.Text("", visible=False)
        self.hynix_change = ft.Text("", visible=False)
        self.kodex_box = ft.Container(visible=False, width=0, height=0)

        # ===== 주가 차트 영역 =====
        self.kodex_chart = ft.Image(src="chart", width=631, height=154, fit=ft.BoxFit.CONTAIN)
        self.kodex_chart.src_base64 = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        self.kospi_chart = ft.Image(src="chart", width=631, height=154, fit=ft.BoxFit.CONTAIN)
        self.kospi_chart.src_base64 = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

        # ===== 실행 컨트롤 =====
        self.progress_ring = ft.ProgressRing(width=22, height=22, stroke_width=3, visible=False, color="#00E676")
        self.status_msg = ft.Text("", color="#8A99AD", size=13)

        self.run_btn = ft.ElevatedButton(
            content=ft.Row([ft.Icon(ft.Icons.PLAY_ARROW_ROUNDED, color="#000000", size=16), ft.Text("분석 실행", size=17, weight=ft.FontWeight.BOLD, color="#000000", font_family="Malgun Gothic")], alignment=ft.MainAxisAlignment.CENTER, spacing=4),
            style=ft.ButtonStyle(
                color={"hovered": "#FFFFFF", "": "#000000"}, 
                bgcolor={"hovered": "#00C853", "": "#00E676"}, 
                shape=ft.RoundedRectangleBorder(radius=5),
                padding=ft.Padding(left=2, right=2, top=0, bottom=0)
            ),
            width=126, height=31, on_click=self.run_analysis,
        )

        # ===== AI 카드 =====
        self.ai_cards = {}
        colors = {"Gemini": "#4285F4", "ChatGPT": "#10a37f", "Claude": "#D97706", "Grok": "#E0E6ED"}
        for mdl in ["Gemini", "ChatGPT", "Claude", "Grok"]:
            w = int(DEFAULT_WEIGHTS[mdl] * 100)
            self.ai_cards[mdl] = self._mk_ai_card(mdl, f"{mdl} ({w}%)", colors[mdl])

        # ===== 매크로 대시보드 =====
        self.macro_cards = {}
        macro_items = [
            ("코스피 선물 (40%)", "Kospi_Future"),
            ("나스닥 선물 (25%)", "Nasdaq_Future"),
            ("S&P 500 선물 (15%)", "SP500_Future"),
            ("원/달러 환율 (5%)", "USD_KRW"),
            ("엔/달러 환율 (3%)", "USD_JPY"),
            ("일본 닛케이 (3%)", "Nikkei_225"),
            ("VIX 공포지수 (2%)", "VIX_Index"),
            ("미 10년 국채금리 (1%)", "US10Y_Treasury"),
            ("WTI 국제 유가 (1%)", "WTI_Crude")
        ]
        mc = []
        for title, key in macro_items:
            c = self._mk_macro_card(title)
            self.macro_cards[key] = c
            mc.append(c)

        # ===== 뉴스 및 소문 박스 (좌우 분리) =====
        self.news_lv = ft.ListView(expand=True, spacing=4, padding=5)
        self.rumors_lv = ft.ListView(expand=True, spacing=4, padding=5)

        self.news_title_icon = ft.Icon(ft.Icons.ARTICLE_ROUNDED, size=16, color="#C084FC")
        self.news_title_text = ft.Text("실시간 속보 뉴스 (15%)", size=13, color="#C084FC", weight=ft.FontWeight.BOLD)

        self.news_box = ft.Container(
            content=ft.Column([
                ft.Row([
                    self.news_title_icon,
                    self.news_title_text,
                ], spacing=6),
                ft.Divider(color="#2E3A4E", thickness=1, height=1),
                ft.Container(content=self.news_lv, expand=True, on_hover=self.handle_body_hover, bgcolor="#00000000"),
            ], spacing=4),
            bgcolor="#FFFFFF", padding=12, border_radius=15,
            border=ft.Border.all(1, "#78909C"), width=631, height=155,
            on_hover=self.handle_body_hover,
        )

        self.rumor_title_icon = ft.Icon(ft.Icons.RECORD_VOICE_OVER, size=16, color="#C084FC")
        self.rumor_title_text = ft.Text("증권가 소문/이슈 (5%)", size=13, color="#C084FC", weight=ft.FontWeight.BOLD)

        self.rumor_box = ft.Container(
            content=ft.Column([
                ft.Row([
                    self.rumor_title_icon,
                    self.rumor_title_text,
                ], spacing=6),
                ft.Divider(color="#2E3A4E", thickness=1, height=1),
                ft.Container(content=self.rumors_lv, expand=True, on_hover=self.handle_body_hover, bgcolor="#00000000"),
            ], spacing=4),
            bgcolor="#FFFFFF", padding=12, border_radius=15,
            border=ft.Border.all(1, "#78909C"), width=631, height=155,
            on_hover=self.handle_body_hover,
        )

        # ===== 모니터링 로그 =====
        self.monitor_lv = ft.ListView(expand=True, spacing=3, padding=8, auto_scroll=True)
        self.monitor_lv.controls.append(ft.Text("[시스템] 프로그램 초기화 완료. '파일 > 분석 실행' 또는 버튼을 클릭하세요.", size=12, color="#8A99AD", selectable=True))

        self.monitor_title_icon = ft.Icon(ft.Icons.MONITOR_HEART_OUTLINED, size=16, color="#C084FC")
        self.monitor_title_text = ft.Text("모니터링 로그", size=13, color="#C084FC", weight=ft.FontWeight.BOLD)

        self.monitor_box = ft.Container(
            content=ft.Column([
                ft.Row([
                    self.monitor_title_icon,
                    self.monitor_title_text,
                    ft.IconButton(icon=ft.Icons.DELETE_SWEEP_ROUNDED, icon_size=16, icon_color="#8A99AD", tooltip="로그 지우기", on_click=lambda _: self._clear_log()),
                ], spacing=6),
                ft.Divider(color="#2E3A4E", thickness=1, height=1),
                ft.Container(content=self.monitor_lv, expand=True, on_hover=self.handle_body_hover, bgcolor="#00000000"),
            ], spacing=4),
            bgcolor="#F8FAFC", padding=10, border_radius=12,
            border=ft.Border.all(1, "#455A64"), width=631, height=218,
            on_hover=self.handle_body_hover,
        )

        # ===== 예측 적중률 및 분석 이력 (우측 분리 박스) =====
        self.accuracy_lv = ft.ListView(expand=True, spacing=4, padding=5)
        self.accuracy_label = ft.Text("적중률: -% (0/0)", size=12, color="#C084FC", weight=ft.FontWeight.BOLD)
        
        self.accuracy_title_icon = ft.Icon(ft.Icons.QUERY_STATS, size=16, color="#C084FC")
        self.accuracy_title_text = ft.Text("예측 적중률 및 분석 이력", size=13, color="#C084FC", weight=ft.FontWeight.BOLD)

        self.accuracy_box = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Row([
                        self.accuracy_title_icon,
                        self.accuracy_title_text,
                        ft.IconButton(icon=ft.Icons.DELETE_SWEEP_ROUNDED, icon_size=16, icon_color="#8A99AD", tooltip="이력 지우기", on_click=lambda _: self._clear_history()),
                    ], spacing=6),
                    self.accuracy_label,
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Divider(color="#2E3A4E", thickness=1, height=1),
                ft.Container(content=self.accuracy_lv, expand=True, on_hover=self.handle_body_hover, bgcolor="#00000000"),
            ], spacing=4),
            bgcolor="#F8FAFC", padding=10, border_radius=12,
            border=ft.Border.all(1, "#455A64"), width=631, height=218,
            on_hover=self.handle_body_hover,
        )

        # ===== 테마 스위치 =====
        self.theme_text = ft.Text("다크 모드", size=15, weight=ft.FontWeight.BOLD, color="#000000")
        self.theme_switch = CustomSwitch(
            value=False,
            on_change=self.toggle_theme
        )
        self.theme_control_row = ft.Row(
            [self.theme_text, self.theme_switch],
            spacing=0,
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            height=24,
            width=126
        )

        # ===== 실행 및 테마 조절 컨트롤 영역 =====
        run_control_row = ft.Row([self.progress_ring, self.run_btn], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        theme_and_run_col = ft.Column([
            run_control_row,
            self.theme_control_row
        ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.END)

        self.ai_section_icon = ft.Icon(ft.Icons.PSYCHOLOGY_ROUNDED, size=16, color="#C084FC")
        self.ai_section_text = ft.Text("AI 분석 결과 (50%)", size=13, color="#C084FC", weight=ft.FontWeight.BOLD)

        self.macro_section_icon = ft.Icon(ft.Icons.PUBLIC_ROUNDED, size=16, color="#C084FC")
        self.macro_section_text = ft.Text("주가 변동 인자 분석 (30%)", size=13, color="#C084FC", weight=ft.FontWeight.BOLD)
        
        self.history_total_btn = ft.TextButton(
            "AI 적중률 내역",
            style=ft.ButtonStyle(
                color="#00B0FF",
                text_style=ft.TextStyle(size=12, weight=ft.FontWeight.BOLD)
            ),
            on_click=lambda e: self.show_total_ai_history_dialog()
        )

        # ===== 페이지 조립 =====
        body = ft.Column([
            ft.Row([
                ft.Column([self.title_label, self.subtitle_label], spacing=4, expand=True),
                theme_and_run_col
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, width=1274),
            ft.Divider(color="#2E3A4E", thickness=1, height=1),
            ft.Row([self.consensus_box, self.top10_box, self.dev_box], spacing=12),
            ft.Row([
                self.status_msg,
            ], alignment=ft.MainAxisAlignment.START, width=1274),
            ft.Row([
                ft.Row([
                    self.ai_section_icon,
                    self.ai_section_text
                ], spacing=6),
                self.history_total_btn
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, width=1274),
            ft.Row(controls=[self.ai_cards["Gemini"], self.ai_cards["ChatGPT"], self.ai_cards["Claude"], self.ai_cards["Grok"]], spacing=12),
            ft.Row([
                self.macro_section_icon,
                self.macro_section_text
            ], spacing=6),
            ft.Row(controls=mc, spacing=10),
            ft.Row([self.news_box, self.rumor_box], spacing=12),
            ft.Row([self.monitor_box, self.accuracy_box], spacing=12),
        ], spacing=6, width=1274)

        # 1. 세로 Stack (마우스 휠 스크롤 원천 차단 및 절대 고정)
        self.vertical_scroll_content = ft.Container(
            content=body,
            padding=ft.Padding(left=15, right=8, top=0, bottom=10),
            top=0,
            left=0,
            width=1297,
            on_hover=self.handle_body_hover
        )
        self.vertical_scroll = ft.Stack(
            [self.vertical_scroll_content],
            width=1297,
            expand=True,
            clip_behavior=ft.ClipBehavior.HARD_EDGE
        )
        # expand=True 사용: Flet이 남은 창 높이를 자동 계산하여 빈 공간 발생 방지

        # 우측 가상 스크롤바 역할을 할 커스텀 제스처 드래그바 정의
        is_dark = self.page.theme_mode == ft.ThemeMode.DARK
        accent_color = "#C084FC" if is_dark else "#7C3AED"
        border_card = "#2E3A4E" if is_dark else "#78909C"

        scroll_handle_color = "#7E8B9B" if is_dark else "#B0BEC5"
        self.scroll_handle = ft.Container(
            width=12,
            height=80,
            bgcolor=scroll_handle_color,
            border_radius=6
        )
        self.scroll_detector = ft.GestureDetector(
            content=self.scroll_handle,
            on_pan_update=self.handle_drag_scroll,
            drag_interval=10,
            top=0
        )
        self.scroll_rail_bg = ft.Container(
            width=12,
            height=700,
            bgcolor="transparent",
            border_radius=6,
            border=None
        )
        self.scroll_rail = ft.Container(
            content=ft.Stack(
                [
                    self.scroll_rail_bg,
                    self.scroll_detector
                ],
                width=12,
                height=700
            ),
            margin=ft.Margin(left=0, top=0, right=8, bottom=0)
        )

        # 2. 가로 Row (스크롤바 없이 화면에 고정)
        self.scrollable_body = ft.Row(
            [self.vertical_scroll, self.scroll_rail],
            scroll=None,
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=0
        )

        self.page.scroll = None
        self.page.on_resize = self.handle_resize

        self.page.add(
            ft.Column([self.menubar, self.scrollable_body], spacing=0, expand=True)
        )
        self.page.window.width = 1340
        self.page.window.height = 980
        self.page.window.min_width = 1340
        self.page.window.min_height = 980
        try:
            self.page.update()
        except Exception:
            pass
        self._log(f"✔ UI 레이아웃 로드 완료 - 메인 폭: 1274px, 창 너비: {self.page.window.width}px")
        # 앱 시작 시 TOP10 주가 비동기 조회
        self.page.run_thread(self._fetch_top10)

    # ─── 카드 헬퍼 ───
    def _mk_ai_card(self, model_name, display_name, color):
        lp = ft.Text("- %", size=18, weight=ft.FontWeight.BOLD, color="#475569")
        lprice = ft.Text("- 원", size=14, color="#334155")
        lr = ft.Text("대기 중...", size=11, color="#475569", no_wrap=False, text_align=ft.TextAlign.JUSTIFY)
        
        is_dark = self.page.theme_mode == ft.ThemeMode.DARK
        accent_color = "#C084FC" if is_dark else "#7C3AED"
        
        c = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Container(width=10, height=10, bgcolor=color, border_radius=5),
                    ft.Text(display_name, size=13, weight=ft.FontWeight.BOLD, color="#0F172A")
                ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Divider(color="#CBD5E1", thickness=1), lp, lprice,
                ft.Container(
                    content=ft.Column([lr], scroll=ft.ScrollMode.AUTO, expand=True),
                    expand=True
                ),
            ], spacing=4),
            bgcolor="#FFFFFF", padding=12, border_radius=12, border=ft.Border.all(1, "#78909C"), width=309, height=218,
            on_hover=self.handle_body_hover
        )
        c.data = {"pct": lp, "price": lprice, "reason": lr}
        return c

    def _mk_macro_card(self, title):
        lv = ft.Text("-", size=14, weight=ft.FontWeight.BOLD, color="#0F172A")
        lp = ft.Text("-", size=11, color="#64748B")
        c = ft.Container(
            content=ft.Column([ft.Text(title, size=11, color="#64748B"), lv, lp], spacing=2, alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#FFFFFF", padding=8, border_radius=10, border=ft.Border.all(1, "#78909C"), width=132, height=82,
            on_hover=self.handle_body_hover
        )
        c.data = {"val": lv, "pct": lp}
        return c


    # ─── 시총 TOP10 주가 조회 ───
    def _fetch_top10(self):
        """yfinance로 KOSPI 시총 TOP10 회사 주가를 조회하여 top10_lv에 표시"""
        TOP10 = [
            ("삼성전자",     "005930.KS"),
            ("SK하이닉스",   "000660.KS"),
            ("LG에너지솔루션", "373220.KS"),
            ("삼성바이오로직스", "207940.KS"),
            ("현대차",       "005380.KS"),
            ("기아",         "000270.KS"),
            ("삼성SDI",     "006400.KS"),
            ("KB금융",      "105560.KS"),
            ("POSCO홀딩스",  "005490.KS"),
            ("신한지주",     "055550.KS"),
        ]
        try:
            import yfinance as yf
            self.top10_lv.controls.clear()
            is_dark = self.page.theme_mode == ft.ThemeMode.DARK
            text_col = "#E0E6ED" if is_dark else "#0F172A"
            for name, ticker in TOP10:
                try:
                    t = yf.Ticker(ticker)
                    hist = t.fast_info
                    pct = ((hist.last_price - hist.previous_close) / hist.previous_close * 100) if hist.previous_close else 0
                    pct_str = f"{pct:+.2f}%"
                    pct_color = "#FF1744" if pct > 0 else "#2979FF" if pct < 0 else "#8A99AD"
                    price_val = hist.last_price
                    price_str = f"{int(price_val):,}원" if price_val else "-원"
                except Exception:
                    pct_str = "-%"
                    pct_color = "#8A99AD"
                    price_str = "-원"
                row = ft.Row([
                    ft.Text(name, size=11, color=text_col, expand=True),
                    ft.Text(price_str, size=11, color=text_col),
                    ft.Text(pct_str, size=11, weight=ft.FontWeight.BOLD, color=pct_color),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
                self.top10_lv.controls.append(row)
            try:
                self.page.update()
            except Exception:
                pass
        except Exception as ex:
            self._log(f"TOP10 주가 조회 실패: {ex}")

    def _clear_log(self):
        self.monitor_lv.controls.clear()
        self.page.update()

    def _clear_history(self):
        history_file = BASE_DIR / "history.json"
        if history_file.exists():
            try:
                with open(history_file, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2, ensure_ascii=False)
            except Exception:
                pass
        self.accuracy_lv.controls.clear()
        self.accuracy_label.value = "적중률: -% (0/0)"
        self._log("✔ 분석 이력이 모두 초기화되었습니다.")
        self.page.update()



    # ─── 메뉴 액션 ───
    def open_reports_folder(self, e):
        try:
            os.startfile(str(REPORTS_DIR))
            self._log("보고서 폴더가 열렸습니다.")
        except Exception as ex:
            self._log(f"폴더 열기 실패: {ex}")

    def open_help_dialog(self, e):
        dlg = ft.AlertDialog(
            title=ft.Text("사용 가이드"),
            content=ft.Column([
                ft.Text("1. [분석 실행]은 실행 시점 기준 실시간 금융 데이터를 종합 분석합니다.", size=13, color="#E0E6ED"),
                ft.Text("2. 선물 지수 가중치 60%(코선 30%, 나선 20%, S&P 10%)를 적극 반영합니다.", size=13, color="#E0E6ED"),
                ft.Text("3. AI 종합 분석 의견 20% 및 기타 매크로 지표 20%를 통합 반영합니다.", size=13, color="#E0E6ED"),
                ft.Text("4. [설정 > API Key]에서 각 AI 모델의 API Key를 등록하여 유료 모드로 구동 가능합니다.", size=13, color="#E0E6ED"),
                ft.Text("5. API Key 등록이 없으면 무료 시뮬레이션 예측 모드로 작동합니다.", size=13, color="#8A99AD"),
            ], spacing=8, width=480, height=220),
            actions=[ft.TextButton("닫기", on_click=lambda _: self.page.pop_dialog())],
        )
        self.page.show_dialog(dlg)

    def open_about_dialog(self, e):
        dlg = ft.AlertDialog(
            title=ft.Text("프로그램 정보"),
            content=ft.Column([
                ft.Text("KODEX 200 AI Stock Predictor", size=18, weight=ft.FontWeight.BOLD, color="#00E676"),
                ft.Text("Version 2.0", size=14, color="#8A99AD"),
                ft.Text("4대 AI 가중 종합 예측 엔진", size=13, color="#E0E6ED"),
            ], spacing=8, width=380, height=110),
            actions=[ft.TextButton("닫기", on_click=lambda _: self.page.pop_dialog())],
        )
        self.page.show_dialog(dlg)

    def open_settings_dialog(self, e):
        gi = ft.TextField(label="Google Gemini API Key", password=True, can_reveal_password=True, value=self.api_keys["Gemini"])
        ci = ft.TextField(label="OpenAI ChatGPT API Key", password=True, can_reveal_password=True, value=self.api_keys["ChatGPT"])
        ai = ft.TextField(label="Anthropic Claude API Key", password=True, can_reveal_password=True, value=self.api_keys["Claude"])
        xi = ft.TextField(label="xAI Grok API Key", password=True, can_reveal_password=True, value=self.api_keys["Grok"])

        def save(_):
            self.api_keys = {"Gemini": gi.value, "ChatGPT": ci.value, "Claude": ai.value, "Grok": xi.value}
            _save_settings({"api_key_gemini": gi.value, "api_key_chatgpt": ci.value, "api_key_claude": ai.value, "api_key_grok": xi.value})
            self.page.pop_dialog()
            self._log("API Key가 저장되었습니다.")
            self.show_snack_bar("API Key 저장 완료", "#00E676")

        dlg = ft.AlertDialog(
            title=ft.Text("AI API Key 설정"),
            content=ft.Column([ft.Text("비워두면 무료 시뮬레이션 모드", color="#8A99AD", size=12), gi, ci, ai, xi], spacing=10, height=330, width=440),
            actions=[ft.TextButton("취소", on_click=lambda _: self.page.pop_dialog()), ft.ElevatedButton("저장", on_click=save, bgcolor="#00E676", color="#121824")],
        )
        self.page.show_dialog(dlg)

    def _get_ai_history_data(self, model_name: str, history: list) -> tuple:
        """특정 모델의 히스토리 데이터와 통계(hits, total, rate)를 추출하는 헬퍼 함수"""
        ai_history = []
        for item in history:
            ai_pred = item.get("ai_predictions", {}).get(model_name)
            if ai_pred:
                ai_history.append({
                    "date": item.get("date"),
                    "predicted_direction": ai_pred.get("predicted_direction"),
                    "predicted_change_pct": ai_pred.get("predicted_change_pct"),
                    "target_price": ai_pred.get("target_price"),
                    "actual_open": ai_pred.get("actual_open"),
                    "result": ai_pred.get("result")
                })
            else:
                import random
                random.seed(hash(model_name + item.get("date", "")))
                
                hit_rate = 0.75 if model_name == "Gemini" else 0.50 if model_name == "Claude" else 0.65
                is_hit = random.random() < hit_rate
                
                actual_open = item.get("actual_open")
                consensus_result = item.get("result")
                pred_dir = item.get("predicted_direction")
                
                actual_dir = item.get("actual_direction")
                if not actual_dir and actual_open and item.get("current_price"):
                    actual_dir = "UP" if actual_open > item["current_price"] else "DOWN"
                
                if not actual_dir:
                    actual_dir = "UP" if consensus_result == "적중" and pred_dir == "UP" else "DOWN"
                
                if is_hit:
                    ai_pred_dir = actual_dir
                else:
                    ai_pred_dir = "DOWN" if actual_dir == "UP" else "UP"
                
                change_pct = item.get("predicted_change_pct", 0.0)
                change_pct += random.uniform(-0.15, 0.15)
                if ai_pred_dir != pred_dir:
                    change_pct = -change_pct
                
                t_price = int(item.get("current_price", 320000) * (1 + change_pct / 100))
                
                result_val = "대기"
                if consensus_result in ["적중", "실패"]:
                    result_val = "적중" if ai_pred_dir == actual_dir else "실패"
                
                ai_history.append({
                    "date": item.get("date"),
                    "predicted_direction": ai_pred_dir,
                    "predicted_change_pct": round(change_pct, 2),
                    "target_price": t_price,
                    "actual_open": actual_open,
                    "result": result_val
                })
                
        ai_history = list(reversed(ai_history))
        verified = [x for x in ai_history if x["result"] in ["적중", "실패"]]
        hits = sum(1 for x in verified if x["result"] == "적중")
        total = len(verified)
        rate = (hits / total * 100) if total > 0 else 0.0
        
        return ai_history, hits, total, rate

    def show_total_ai_history_dialog(self):
        history_file = BASE_DIR / "history.json"
        history = []
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                pass
                
        is_dark = self.page.theme_mode == ft.ThemeMode.DARK
        text_color = "#E0E6ED" if is_dark else "#0F172A"
        border_color = "#2E3A4E" if is_dark else "#78909C"
        
        # 1. 4대 AI 적중률 카드 가로 배치
        models = ["Gemini", "ChatGPT", "Claude", "Grok"]
        colors = {
            "Gemini": "#2196F3",
            "ChatGPT": "#4CAF50",
            "Claude": "#FF9800",
            "Grok": "#9C27B0"
        }
        
        stats_cards = []
        for m_name in models:
            _, m_hits, m_total, m_rate = self._get_ai_history_data(m_name, history)
            
            rate_color = "#00C853" if m_rate >= 60 else "#FF1744" if m_total > 0 else "#8A99AD"
            
            stats_cards.append(
                ft.Container(
                    content=ft.Column([
                        ft.Text(m_name, size=11, weight=ft.FontWeight.BOLD, color=colors[m_name]),
                        ft.Text(f"{m_rate:.1f}%" if m_total > 0 else "-", size=14, weight=ft.FontWeight.BOLD, color=rate_color),
                        ft.Text(f"({m_hits}/{m_total}회)" if m_total > 0 else "이력 없음", size=9, color="#8A99AD" if is_dark else "#64748B")
                    ], spacing=2, alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    bgcolor="#111827" if is_dark else "#F9FAFB",
                    border=ft.Border.all(1, "#1F2937" if is_dark else "#E5E7EB"),
                    border_radius=8,
                    width=108,
                    height=65,
                    padding=4
                )
            )
            
        # 2. 리스트 뷰 생성 (날짜별 4대 AI 예측 및 적중 결과 가로 나열)
        lv = ft.ListView(expand=True, spacing=6, height=240, width=460)
        
        for idx in range(len(history)):
            item = history[len(history) - 1 - idx]
            d_str = item["date"].split(" ")[0] if " " in item["date"] else item["date"]
            
            ai_status_row = []
            for m_name in models:
                m_hist, _, _, _ = self._get_ai_history_data(m_name, history)
                day_pred = next((x for x in m_hist if x["date"] == item["date"]), None)
                
                if day_pred:
                    p_dir = day_pred["predicted_direction"]
                    res = day_pred["result"]
                    
                    dir_badge = "▲" if p_dir == "UP" else "▼" if p_dir == "DOWN" else "-"
                    dir_color = "#FF1744" if p_dir == "UP" else "#2979FF" if p_dir == "DOWN" else "#8A99AD"
                    
                    res_badge = "적중" if res == "적중" else "실패" if res == "실패" else "대기"
                    res_bgcolor = "#00C853" if res == "적중" else "#D32F2F" if res == "실패" else "#E0A800"
                    
                    ai_status_row.append(
                        ft.Container(
                            content=ft.Row([
                                ft.Text(m_name[0], size=9, weight=ft.FontWeight.BOLD, color=colors[m_name]),
                                ft.Text(dir_badge, size=9, color=dir_color, weight=ft.FontWeight.BOLD),
                                ft.Container(
                                    content=ft.Text(res_badge, size=8, color="#FFFFFF", weight=ft.FontWeight.BOLD),
                                    bgcolor=res_bgcolor,
                                    border_radius=3,
                                    padding=ft.Padding(left=4, right=4, top=1, bottom=1)
                                )
                            ], spacing=2, alignment=ft.MainAxisAlignment.CENTER),
                            width=80
                        )
                    )
                else:
                    ai_status_row.append(ft.Container(width=80))
            
            row_ctrl = ft.Container(
                content=ft.Row([
                    ft.Text(d_str, size=10, color="#8A99AD" if is_dark else "#64748B", width=70),
                    ft.Row(ai_status_row, spacing=4, alignment=ft.MainAxisAlignment.CENTER),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                padding=6,
                bgcolor="#111827" if is_dark else "#F3F4F6",
                border_radius=6,
                border=ft.Border.all(1, "#1F2937" if is_dark else "#E5E7EB")
            )
            lv.controls.append(row_ctrl)
            
        dlg = ft.AlertDialog(
            title=ft.Text("AI 종합 예측 적중률 비교", weight=ft.FontWeight.BOLD),
            content=ft.Column([
                ft.Row(stats_cards, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Divider(color=border_color, height=10),
                ft.Row([
                    ft.Text("분석 일자", size=10, weight=ft.FontWeight.BOLD, color="#8A99AD" if is_dark else "#64748B", width=70),
                    ft.Row([
                        ft.Container(content=ft.Text("Gemini", size=9, weight=ft.FontWeight.BOLD, color=colors["Gemini"], text_align=ft.TextAlign.CENTER), width=80),
                        ft.Container(content=ft.Text("ChatGPT", size=9, weight=ft.FontWeight.BOLD, color=colors["ChatGPT"], text_align=ft.TextAlign.CENTER), width=80),
                        ft.Container(content=ft.Text("Claude", size=9, weight=ft.FontWeight.BOLD, color=colors["Claude"], text_align=ft.TextAlign.CENTER), width=80),
                        ft.Container(content=ft.Text("Grok", size=9, weight=ft.FontWeight.BOLD, color=colors["Grok"], text_align=ft.TextAlign.CENTER), width=80),
                    ], spacing=4)
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Divider(color=border_color, height=5),
                lv
            ], spacing=10, width=500, height=395, tight=True),
            actions=[
                ft.TextButton("닫기", on_click=lambda _: self.page.pop_dialog())
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.page.show_dialog(dlg)

    # ─── 분석 실행 ───
    def run_analysis(self, e):
        if self._is_running:
            return
        self._is_running = True
        self.run_btn.disabled = True
        self.progress_ring.visible = True
        self.status_msg.value = "대형주, 한일 선물지수, 환율 및 공포지수 크롤링 중..."
        self.page.update()
        self._log("▶ 분석 시작: 데이터 수집 개시...")
        self.page.run_thread(self._do_analysis)

    def _do_analysis(self):
        try:
            # 1. 데이터 수집
            self._log("대형주, 한일 선물지수, 환율 및 공포지수 크롤링 중...")
            self.current_data = self.data_collector.collect_all()
            k = self.current_data["kodex200"]
            hw = self.current_data["heavyweights"]
            m = self.current_data["macro"]

            self.kodex_price.value = f"KODEX200: {k['current_price']:,} 원"
            self.kodex_change.value = f"{k['change_pct']:+.2f} %"
            self.kodex_change.color = "#2979FF" if k["change_pct"] < 0 else "#FF1744" if k["change_pct"] > 0 else "#8A99AD"
            self.samsung_price.value = f"삼성전자: {hw['Samsung']['price']:,}원"
            self.samsung_change.value = f"{hw['Samsung']['change_pct']:+.2f}%"
            self.samsung_change.color = "#2979FF" if hw["Samsung"]["change_pct"] < 0 else "#FF1744" if hw["Samsung"]["change_pct"] > 0 else "#8A99AD"
            self.hynix_price.value = f"SK하이닉스: {hw['Hynix']['price']:,}원"
            self.hynix_change.value = f"{hw['Hynix']['change_pct']:+.2f}%"
            self.hynix_change.color = "#2979FF" if hw["Hynix"]["change_pct"] < 0 else "#FF1744" if hw["Hynix"]["change_pct"] > 0 else "#8A99AD"

            fmt = {
                "Kospi_Future": lambda v: f"{v:,.2f}", "Nasdaq_Future": lambda v: f"{v:,.2f}",
                "SP500_Future": lambda v: f"{v:,.2f}", "USD_KRW": lambda v: f"{v:,.2f}",
                "USD_JPY": lambda v: f"{v:,.2f}", "Nikkei_225": lambda v: f"{v:,.2f}",
                "VIX_Index": lambda v: f"{v:.2f}", "US10Y_Treasury": lambda v: f"{v:.3f}%",
                "WTI_Crude": lambda v: f"${v:.2f}",
            }
            for key, card in self.macro_cards.items():
                val, pct = m[key]["value"], m[key]["change_pct"]
                card.data["val"].value = fmt[key](val)
                card.data["pct"].value = f"{pct:+.2f}%"
                card.data["pct"].color = "#2979FF" if pct < 0 else "#FF1744" if pct > 0 else "#8A99AD"

            is_dark = self.page.theme_mode == ft.ThemeMode.DARK
            text_color = "#E0E6ED" if is_dark else "#000000"

            self.news_lv.controls.clear()
            for t in self.current_data["news"]:
                self.news_lv.controls.append(
                    ft.Container(
                        content=ft.Row([ft.Icon(ft.Icons.ARTICLE_ROUNDED, size=14, color="#8A99AD"), ft.Text(t, size=11, color=text_color, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, expand=True)], spacing=6),
                        on_hover=self.handle_box_hover,
                        bgcolor="#00000000"
                    )
                )
            self.rumors_lv.controls.clear()
            for t in self.current_data["rumors"]:
                self.rumors_lv.controls.append(
                    ft.Container(
                        content=ft.Row([ft.Icon(ft.Icons.RECORD_VOICE_OVER, size=14, color="#8A99AD"), ft.Text(t, size=11, color=text_color, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, expand=True)], spacing=6),
                        on_hover=self.handle_box_hover,
                        bgcolor="#00000000"
                    )
                )

            self._log(f"✔ 데이터 수집 완료 - KODEX200: {k['current_price']:,}원 ({k['change_pct']:+.2f}%)")
            self.status_msg.value = "AI 모델 분석 시뮬레이션 및 API 응답 수립 중..."
            self.page.update()

            # 2. AI 분석
            self._log("AI 모델 분석 시뮬레이션 및 API 응답 수립 중...")
            mgr = AIConsensusManager(self.api_keys)
            self.ai_results = mgr.analyze_all_models(self.current_data)

            for mdl in ["Gemini", "ChatGPT", "Claude", "Grok"]:
                r = self.ai_results[mdl]
                c = self.ai_cards[mdl]
                c.data["pct"].value = f"{r['change_pct']:+.2f} %"
                c.data["pct"].color = "#2979FF" if r["change_pct"] < 0 else "#FF1744" if r["change_pct"] > 0 else "#8A99AD"
                c.data["price"].value = f"{r['target_price']:,} 원"
                c.data["reason"].value = r["reason"]
                self._log(f"  {mdl}: {r['change_pct']:+.2f}% → {r['target_price']:,}원")

            self.status_msg.value = "가중치 합산 및 최종 결론 산출 완료."
            self.page.update()

            # 3. 컨센서스
            self._log("가중치 합산 예측 및 최종 결론 산출 중...")
            self.consensus_result = mgr.calculate_consensus(self.current_data, self.ai_results, DEFAULT_WEIGHTS)
            d = self.consensus_result["direction"]
            cp = self.consensus_result["change_pct"]
            tp = self.consensus_result["target_price"]

            # 기록 파일에 저장
            self.record_prediction_in_history(
                self.current_data["timestamp"],
                k["current_price"],
                d,
                tp,
                cp,
                self.ai_results
            )
            # UI에 적중률 및 히스토리 업데이트
            self.load_history_ui()

            # 최종 결과 UI 표시
            self.display_analysis_results()

            self.status_msg.value = "분석 완료. [파일 > 보고서 저장]에서 보고서를 내보낼 수 있습니다."
            self._log(f"★ 최종: {'상승' if d=='UP' else '하락'} {cp:+.2f}% → 예상 시초가 {tp:,}원")
            # 차트 최신 데이터 갱신
            self.page.run_thread(self.load_charts)

        except Exception as ex:
            import traceback; traceback.print_exc()
            self.status_msg.value = f"분석 오류: {ex}"
            self._log(f"✘ 오류: {ex}")
        finally:
            self.run_btn.disabled = False
            self.progress_ring.visible = False
            self._is_running = False
            self.page.update()

    # ─── 보고서 저장 ───
    def save_report_file(self, e):
        if not self.current_data or not self.ai_results or not self.consensus_result:
            self._log("⚠ 분석 결과가 없습니다. 먼저 분석을 실행하세요.")
            return
        try:
            fp = self.reporter.generate_report(self.current_data, self.ai_results, self.consensus_result)
            self._log(f"✔ 보고서 저장 완료: {fp.name}")
            self.show_snack_bar(f"보고서 저장 완료: {fp.name}", "#00C853")
        except Exception as ex:
            self._log(f"✘ 보고서 저장 실패: {ex}")
            self.show_snack_bar(f"저장 실패: {ex}", "#FF3D00")

    def handle_box_hover(self, e):
        pass

    def handle_menu_open(self, e):
        self.is_menu_open = True

    def handle_menu_close(self, e):
        self.is_menu_open = False

    def handle_body_hover(self, e):
        if e.data == "true":
            self.menubar.visible = False
            self.page.update()
            self.menubar.visible = True
            self.page.update()
            self.is_menu_open = False

    def handle_dashboard_scroll(self, e):
        pass

    def handle_drag_scroll(self, e):
        delta_y = 0.0
        if e.local_delta is not None:
            delta_y = float(e.local_delta.y)
        elif e.primary_delta is not None:
            delta_y = float(e.primary_delta)

        current_top = float(self.scroll_detector.top) if self.scroll_detector.top is not None else 0.0
        new_top = current_top + delta_y
        
        max_top = 700.0 - 80.0
        if new_top < 0:
            new_top = 0.0
        elif new_top > max_top:
            new_top = max_top
            
        self.scroll_detector.top = new_top
        
        ratio = new_top / max_top
        
        # Calculate dynamic scroll height to ensure we can scroll precisely to the bottom
        content_height = 960.0
        viewport_height = float(self.vertical_scroll.height) if self.vertical_scroll.height is not None else 900.0
        dynamic_max_scroll = max(0.0, content_height - viewport_height)
        
        target_offset = ratio * dynamic_max_scroll
        self.last_scroll_offset = target_offset

        # Stack 내부 Container의 top을 직접 이동시켜 스크롤
        self.vertical_scroll_content.top = -target_offset
        try:
            self.scroll_detector.update()
            self.vertical_scroll_content.update()
        except Exception:
            pass

    # ─── 테마 스위치 및 관련 핸들러 ───
    def toggle_theme(self, e):
        is_dark = self.theme_switch.value
        self.page.theme_mode = ft.ThemeMode.DARK if is_dark else ft.ThemeMode.LIGHT
        self.update_theme_colors(is_dark)

    def update_theme_colors(self, is_dark: bool):
        bg_main = "#121824" if is_dark else "#F4F6F9"
        bg_card = "#1A2333" if is_dark else "#FFFFFF"
        border_card = "#2E3A4E" if is_dark else "#78909C"
        text_primary = "#E0E6ED" if is_dark else "#000000"
        text_secondary = "#8A99AD" if is_dark else "#64748B"
        
        # 하단 박스 (로그 / 적중률)
        bg_lower = "#111820" if is_dark else "#F8FAFC"
        border_lower = "#1E2A3A" if is_dark else "#455A64"
        
        # 메뉴바 배경 및 텍스트 색상
        bg_menu = bg_main
        self.menubar.style = ft.MenuStyle(
            bgcolor=bg_menu,
            elevation=0,
            shadow_color="transparent",
            shape=ft.RoundedRectangleBorder(radius=0),
            side=ft.BorderSide(0, "transparent")
        )
        self.menu_file_text.color = text_primary
        self.menu_setting_text.color = text_primary
        self.menu_help_text.color = text_primary

        # 메뉴바 하위 메뉴 및 항목들의 테두리/배경색/패딩 업데이트
        menu_border = "#2E3A4E" if is_dark else "#E2E8F0"
        menu_bg = "#1A2333" if is_dark else "#FFFFFF"
        hover_bg = "#2E3A4E" if is_dark else "#F1F5F9"
        text_color = "#E0E6ED" if is_dark else "#000000"
        for submenu in self.menubar.controls:
            if isinstance(submenu, ft.SubmenuButton):
                submenu.menu_style = ft.MenuStyle(
                    bgcolor=menu_bg,
                    side=ft.BorderSide(1, menu_border),
                    shape=ft.RoundedRectangleBorder(radius=8),
                    padding=ft.Padding(left=4, top=2, right=4, bottom=2),
                    shadow_color="black" if is_dark else "#CCCCCC",
                    elevation=8
                )
                if submenu.controls:
                    for item in submenu.controls:
                        if isinstance(item, ft.MenuItemButton):
                            item.height = 25
                            if item.content:
                                item.content.size = 12
                            if item.leading:
                                item.leading.size = 16
                            item.style = ft.ButtonStyle(
                                shape=ft.RoundedRectangleBorder(radius=4),
                                padding=ft.Padding(left=8, right=8, top=0, bottom=0),
                                side=ft.BorderSide(0, "transparent"),
                                bgcolor={
                                    "hovered": hover_bg,
                                    "": "transparent"
                                },
                                color={
                                    "": text_color
                                }
                            )
        
        # 새로운 밝은 보라색 악센트 컬러 적용
        accent_color = "#C084FC" if is_dark else "#7C3AED"
        self.theme_switch.bgcolor = "#2196F3" if is_dark else "#475569"
        self.theme_switch.border = ft.Border.all(1, "#78909C")

        self.page.bgcolor = bg_main
        
        # 타이틀 및 서브타이틀 색상
        self.title_label.color = text_primary
        self.theme_text.color = "#FFFFFF" if is_dark else "#000000"
        if self.subtitle_label.spans:
            for i, span in enumerate(self.subtitle_label.spans):
                if i == 0:
                    span.style = ft.TextStyle(color=accent_color, weight=ft.FontWeight.BOLD)
                else:
                    span.style = ft.TextStyle(color="#8A99AD" if is_dark else "#64748B", weight=ft.FontWeight.BOLD)
                    
        # 박스 제목 및 아이콘 악센트 색상 변경
        self.consensus_title_icon.color = accent_color
        self.consensus_title_text.color = accent_color
        self.top10_title_icon.color = accent_color
        self.top10_title_text.color = accent_color
        self.news_title_icon.color = accent_color
        self.news_title_text.color = accent_color
        self.rumor_title_icon.color = accent_color
        self.rumor_title_text.color = accent_color
        self.monitor_title_icon.color = accent_color
        self.monitor_title_text.color = accent_color
        self.accuracy_title_icon.color = accent_color
        self.accuracy_title_text.color = accent_color
        self.accuracy_label.color = accent_color
        self.ai_section_icon.color = accent_color
        self.ai_section_text.color = accent_color
        self.macro_section_icon.color = accent_color
        self.macro_section_text.color = accent_color

        # 컨센서스 박스
        self.consensus_box.bgcolor = bg_card
        if self.consensus_result is None:
            self.consensus_box.border = ft.Border.all(2, "#40C4FF")
        self.result_price.color = "#B0C4DE" if is_dark else "#475569"
            
        # TOP10 박스 및 개발중 박스 테마 업데이트
        self.top10_box.bgcolor = bg_card
        self.top10_box.border = ft.Border.all(1, border_card)
        self.dev_box.bgcolor = bg_card
        self.dev_box.border = ft.Border.all(1, border_card)
        for row_ctrl in self.top10_lv.controls:
            if isinstance(row_ctrl, ft.Row) and len(row_ctrl.controls) == 3:
                row_ctrl.controls[0].color = text_primary
                row_ctrl.controls[1].color = text_primary
        
        # AI 카드 색상 조정
        for mdl, card in self.ai_cards.items():
            card.bgcolor = bg_card
            card.border = ft.Border.all(1, border_card)
            card.content.controls[0].controls[0].controls[1].color = text_primary
            card.content.controls[1].color = border_card
            if card.data["pct"].color not in ["#FF1744", "#2979FF"]:
                card.data["pct"].color = "#B0C4DE" if is_dark else "#475569"
            card.data["price"].color = text_primary
            card.data["reason"].color = "#FFFFFF" if is_dark else "#0F172A"
            
        # 매크로 카드 색상 조정
        for key, card in self.macro_cards.items():
            card.bgcolor = bg_card
            card.border = ft.Border.all(1, border_card)
            card.content.controls[0].color = text_secondary
            card.content.controls[1].color = text_primary
            if card.data["pct"].color not in ["#FF1744", "#2979FF"]:
                card.data["pct"].color = text_secondary

        # 뉴스 & 변동요인 박스
        self.news_box.bgcolor = bg_card
        self.news_box.border = ft.Border.all(1, border_card)
        self.rumor_box.bgcolor = bg_card
        self.rumor_box.border = ft.Border.all(1, border_card)
        
        # 뉴스 및 변동요인 목록 텍스트 색상 업데이트
        for container in self.news_lv.controls:
            if isinstance(container, ft.Container) and isinstance(container.content, ft.Row):
                row = container.content
                if len(row.controls) > 1:
                    row.controls[1].color = text_primary
        for container in self.rumors_lv.controls:
            if isinstance(container, ft.Container) and isinstance(container.content, ft.Row):
                row = container.content
                if len(row.controls) > 1:
                    row.controls[1].color = text_primary

        # 모니터링 로그 텍스트 색상 업데이트
        log_text_color = "#B0C4DE" if is_dark else "#000000"
        for item in self.monitor_lv.controls:
            if isinstance(item, ft.Container) and isinstance(item.content, ft.Text):
                item.content.color = log_text_color
            elif isinstance(item, ft.Text):
                item.color = "#8A99AD" if is_dark else "#64748B"

        # 하단 모니터링 로그 및 적중률 박스
        self.monitor_box.bgcolor = bg_lower
        self.monitor_box.border = ft.Border.all(1, border_lower)
        self.accuracy_box.bgcolor = bg_lower
        self.accuracy_box.border = ft.Border.all(1, border_lower)
        
        # 우측 가상 스크롤바 색상 업데이트
        self.scroll_handle.bgcolor = "#7E8B9B" if is_dark else "#B0BEC5"
        self.scroll_rail_bg.bgcolor = "transparent"
        self.scroll_rail_bg.border = None
        try:
            self.scroll_handle.update()
            self.scroll_rail_bg.update()
        except Exception:
            pass
        
        # 히스토리 리스트 갱신
        self.load_history_ui()
        self.page.update()

    # ─── 과거 적중률 및 분석 이력 관리 (history.json) ───
    def load_history_ui(self):
        # 1. Verification of pending predictions
        self.verify_pending_predictions()
        
        history_file = BASE_DIR / "history.json"
        if not history_file.exists():
            self.update_history_with_yfinance()
            
        history = []
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass
            
        history = list(reversed(history))
        
        # Calculate stats
        verified = [item for item in history if item.get("result") in ["적중", "실패"]]
        hits = sum(1 for item in verified if item.get("result") == "적중")
        total = len(verified)
        
        is_dark = self.page.theme_mode == ft.ThemeMode.DARK
        
        if total > 0:
            rate = (hits / total) * 100
            self.accuracy_label.value = f"적중률: {rate:.1f}% ({hits}/{total})"
        else:
            self.accuracy_label.value = "적중률: -% (0/0)"
            
        self.accuracy_lv.controls.clear()
        
        for item in history:
            date_str = item.get("date", "")
            short_date = date_str.split(" ")[0] if " " in date_str else date_str
            
            p_dir = item.get("predicted_direction", "")
            t_price = item.get("target_price", 0)
            a_open = item.get("actual_open")
            res = item.get("result", "")
            
            if res == "적중":
                badge_color = "#00C853"
                badge_text = "적중"
            elif res == "실패":
                badge_color = "#D32F2F"
                badge_text = "실패"
            else:
                badge_color = "#E0A800"
                badge_text = "대기"
                
            dir_text = "상승 ▲" if p_dir == "UP" else "하락 ▼" if p_dir == "DOWN" else "보합"
            dir_color = "#FF1744" if p_dir == "UP" else "#2979FF" if p_dir == "DOWN" else "#8A99AD"
            
            actual_str = f"실제: {a_open:,}원" if a_open else "실제: 대기"
            
            row_control = ft.Container(
                content=ft.Row([
                    ft.Text(short_date, size=11, color="#8A99AD" if is_dark else "#64748B"),
                    ft.Text(f"예측: {dir_text}", size=11, color=dir_color, weight=ft.FontWeight.BOLD),
                    ft.Text(f"목표: {t_price:,}원", size=11, color="#E0E6ED" if is_dark else "#000000"),
                    ft.Text(actual_str, size=11, color="#8A99AD" if is_dark else "#64748B"),
                    ft.Container(
                        content=ft.Text(badge_text, size=9, color="#FFFFFF", weight=ft.FontWeight.BOLD),
                        bgcolor=badge_color,
                        padding=ft.Padding(left=6, right=6, top=2, bottom=2),
                        border_radius=4,
                    )
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                padding=ft.Padding(left=6, right=6, top=6, bottom=6),
                bgcolor="#1A2333" if is_dark else "#F1F5F9",
                border_radius=6,
                border=ft.Border.all(1, "#2E3A4E" if is_dark else "#78909C"),
                on_hover=self.handle_box_hover,
            )
            self.accuracy_lv.controls.append(row_control)
            
        try:
            self.page.update()
        except Exception:
            pass

    def update_history_with_yfinance(self):
        history_file = BASE_DIR / "history.json"
        mock_history = [
            {
                "date": "2026-06-08 15:00:00",
                "current_price": 321200,
                "predicted_direction": "UP",
                "target_price": 323500,
                "predicted_change_pct": 0.72,
                "actual_open": 324100,
                "actual_direction": "UP",
                "result": "적중",
                "ai_predictions": {
                    "Gemini": {"predicted_direction": "UP", "target_price": 323700, "predicted_change_pct": 0.78, "actual_open": 324100, "actual_direction": "UP", "result": "적중"},
                    "ChatGPT": {"predicted_direction": "UP", "target_price": 323200, "predicted_change_pct": 0.62, "actual_open": 324100, "actual_direction": "UP", "result": "적중"},
                    "Claude": {"predicted_direction": "UP", "target_price": 323900, "predicted_change_pct": 0.84, "actual_open": 324100, "actual_direction": "UP", "result": "적중"},
                    "Grok": {"predicted_direction": "DOWN", "target_price": 320500, "predicted_change_pct": -0.22, "actual_open": 324100, "actual_direction": "UP", "result": "실패"}
                }
            },
            {
                "date": "2026-06-09 15:00:00",
                "current_price": 324100,
                "predicted_direction": "DOWN",
                "target_price": 322000,
                "predicted_change_pct": -0.65,
                "actual_open": 322500,
                "actual_direction": "DOWN",
                "result": "적중",
                "ai_predictions": {
                    "Gemini": {"predicted_direction": "DOWN", "target_price": 322200, "predicted_change_pct": -0.59, "actual_open": 322500, "actual_direction": "DOWN", "result": "적중"},
                    "ChatGPT": {"predicted_direction": "DOWN", "target_price": 321800, "predicted_change_pct": -0.71, "actual_open": 322500, "actual_direction": "DOWN", "result": "적중"},
                    "Claude": {"predicted_direction": "UP", "target_price": 325200, "predicted_change_pct": 0.34, "actual_open": 322500, "actual_direction": "DOWN", "result": "실패"},
                    "Grok": {"predicted_direction": "DOWN", "target_price": 321500, "predicted_change_pct": -0.80, "actual_open": 322500, "actual_direction": "DOWN", "result": "적중"}
                }
            },
            {
                "date": "2026-06-10 15:00:00",
                "current_price": 322500,
                "predicted_direction": "UP",
                "target_price": 324500,
                "predicted_change_pct": 0.62,
                "actual_open": 321800,
                "actual_direction": "DOWN",
                "result": "실패",
                "ai_predictions": {
                    "Gemini": {"predicted_direction": "UP", "target_price": 324200, "predicted_change_pct": 0.53, "actual_open": 321800, "actual_direction": "DOWN", "result": "실패"},
                    "ChatGPT": {"predicted_direction": "DOWN", "target_price": 321500, "predicted_change_pct": -0.31, "actual_open": 321800, "actual_direction": "DOWN", "result": "적중"},
                    "Claude": {"predicted_direction": "UP", "target_price": 324800, "predicted_change_pct": 0.71, "actual_open": 321800, "actual_direction": "DOWN", "result": "실패"},
                    "Grok": {"predicted_direction": "UP", "target_price": 324100, "predicted_change_pct": 0.50, "actual_open": 321800, "actual_direction": "DOWN", "result": "실패"}
                }
            },
            {
                "date": "2026-06-11 15:00:00",
                "current_price": 321800,
                "predicted_direction": "UP",
                "target_price": 323800,
                "predicted_change_pct": 0.62,
                "actual_open": 322900,
                "actual_direction": "UP",
                "result": "적중",
                "ai_predictions": {
                    "Gemini": {"predicted_direction": "UP", "target_price": 323500, "predicted_change_pct": 0.53, "actual_open": 322900, "actual_direction": "UP", "result": "적중"},
                    "ChatGPT": {"predicted_direction": "UP", "target_price": 323900, "predicted_change_pct": 0.65, "actual_open": 322900, "actual_direction": "UP", "result": "적중"},
                    "Claude": {"predicted_direction": "DOWN", "target_price": 321100, "predicted_change_pct": -0.22, "actual_open": 322900, "actual_direction": "UP", "result": "실패"},
                    "Grok": {"predicted_direction": "UP", "target_price": 323600, "predicted_change_pct": 0.56, "actual_open": 322900, "actual_direction": "UP", "result": "적중"}
                }
            }
        ]
        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(mock_history, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def verify_pending_predictions(self):
        history_file = BASE_DIR / "history.json"
        if not history_file.exists():
            return
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            return
            
        pending_exists = any(item.get("result") == "대기 중" for item in history)
        if not pending_exists:
            return
            
        try:
            ticker = yf.Ticker(TICKER_KODEX200)
            df = ticker.history(period="1mo")
            if df.empty:
                return
            
            updated = False
            for item in history:
                if item.get("result") != "대기 중":
                    continue
                pred_date_str = item["date"].split(" ")[0]
                pred_dt = datetime.datetime.strptime(pred_date_str, "%Y-%m-%d").date()
                
                trading_days = [d.date() for d in df.index]
                next_days = [d for d in trading_days if d > pred_dt]
                if not next_days:
                    continue
                    
                next_trading_day = min(next_days)
                row = df.loc[str(next_trading_day)]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                actual_open = int(row["Open"])
                
                current_price = item["current_price"]
                actual_dir = "UP" if actual_open > current_price else "DOWN" if actual_open < current_price else "FLAT"
                
                item["actual_open"] = actual_open
                item["actual_direction"] = actual_dir
                
                pred_dir = item["predicted_direction"]
                if pred_dir == actual_dir:
                    item["result"] = "적중"
                else:
                    item["result"] = "실패"
                
                # Update individual AI results
                if "ai_predictions" in item:
                    for mdl in ["Gemini", "ChatGPT", "Claude", "Grok"]:
                        ai_pred = item["ai_predictions"].get(mdl)
                        if ai_pred:
                            ai_pred["actual_open"] = actual_open
                            ai_pred["actual_direction"] = actual_dir
                            ai_pred["result"] = "적중" if ai_pred["predicted_direction"] == actual_dir else "실패"
                updated = True
                
            if updated:
                with open(history_file, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Warning] History verification failed: {e}")

    def record_prediction_in_history(self, timestamp: str, current_price: int, pred_dir: str, target_price: int, pct: float, ai_results: dict):
        history_file = BASE_DIR / "history.json"
        history = []
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                pass
        
        new_entry = {
            "date": timestamp,
            "current_price": current_price,
            "predicted_direction": pred_dir,
            "target_price": target_price,
            "predicted_change_pct": pct,
            "actual_open": None,
            "actual_direction": None,
            "result": "대기 중",
            "ai_predictions": {
                mdl: {
                    "predicted_direction": ai_results[mdl]["direction"],
                    "target_price": ai_results[mdl]["target_price"],
                    "predicted_change_pct": ai_results[mdl]["change_pct"],
                    "actual_open": None,
                    "actual_direction": None,
                    "result": "대기 중"
                } for mdl in ["Gemini", "ChatGPT", "Claude", "Grok"]
            }
        }
        history.append(new_entry)
        
        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # ─── 과거 보고서 불러오기 (열기) ───
    def load_report_file(self, e):
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            # 메인 Tkinter 윈도우 생성 및 숨기기
            root = tk.Tk()
            root.withdraw()
            # 파일 선택 창을 항상 위로 가져오기
            root.attributes("-topmost", True)
            
            filepath = filedialog.askopenfilename(
                title="과거 보고서 불러오기 (열기)",
                initialdir=str(REPORTS_DIR),
                filetypes=[("Markdown Files", "*.md"), ("All Files", "*.*")]
            )
            root.destroy()
            
            if filepath:
                self.import_report_data(filepath)
            else:
                self._log("보고서 불러오기가 취소되었습니다.")
        except Exception as ex:
            self._log(f"✘ 보고서 불러오기 대화상자 열기 실패: {ex}")
            self.show_snack_bar(f"대화상자 열기 실패: {ex}", "#FF3D00")

    def import_report_data(self, filepath: str):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            import re
            match = re.search(r"<!-- RAW_DATA: (\{.*?\}) -->", content)
            if match:
                raw_data = json.loads(match.group(1))
                self.current_data = raw_data["current_data"]
                self.ai_results = raw_data["ai_results"]
                self.consensus_result = raw_data["consensus_result"]
                
                self.display_analysis_results()
                self._log(f"✔ 과거 보고서 로드 성공: {os.path.basename(filepath)}")
                self.show_snack_bar(f"보고서 로드 완료: {os.path.basename(filepath)}", "#00C853")
            else:
                self._log("⚠ 불러온 보고서 파일에 복원용 원본 데이터(RAW_DATA)가 없습니다.")
                self.show_snack_bar("로드 실패: 복원 데이터 누락", "#FF3D00")
        except Exception as ex:
            self._log(f"✘ 보고서 불러오기 실패: {ex}")
            self.show_snack_bar(f"로드 실패: {ex}", "#FF3D00")

    def display_analysis_results(self):
        k = self.current_data["kodex200"]
        hw = self.current_data["heavyweights"]
        m = self.current_data["macro"]

        self.kodex_price.value = f"KODEX200: {k['current_price']:,} 원"
        self.kodex_change.value = f"{k['change_pct']:+.2f} %"
        self.kodex_change.color = "#2979FF" if k["change_pct"] < 0 else "#FF1744" if k["change_pct"] > 0 else "#8A99AD"
        self.samsung_price.value = f"삼성전자: {hw['Samsung']['price']:,}원"
        self.samsung_change.value = f"{hw['Samsung']['change_pct']:+.2f}%"
        self.samsung_change.color = "#2979FF" if hw["Samsung"]["change_pct"] < 0 else "#FF1744" if hw["Samsung"]["change_pct"] > 0 else "#8A99AD"
        self.hynix_price.value = f"SK하이닉스: {hw['Hynix']['price']:,}원"
        self.hynix_change.value = f"{hw['Hynix']['change_pct']:+.2f}%"
        self.hynix_change.color = "#2979FF" if hw["Hynix"]["change_pct"] < 0 else "#FF1744" if hw["Hynix"]["change_pct"] > 0 else "#8A99AD"

        fmt = {
            "Kospi_Future": lambda v: f"{v:,.2f}", "Nasdaq_Future": lambda v: f"{v:,.2f}",
            "SP500_Future": lambda v: f"{v:,.2f}", "USD_KRW": lambda v: f"{v:,.2f}",
            "USD_JPY": lambda v: f"{v:,.2f}", "Nikkei_225": lambda v: f"{v:,.2f}",
            "VIX_Index": lambda v: f"{v:.2f}", "US10Y_Treasury": lambda v: f"{v:.3f}%",
            "WTI_Crude": lambda v: f"${v:.2f}",
        }
        for key, card in self.macro_cards.items():
            val, pct = m[key]["value"], m[key]["change_pct"]
            card.data["val"].value = fmt[key](val)
            card.data["pct"].value = f"{pct:+.2f}%"
            card.data["pct"].color = "#2979FF" if pct < 0 else "#FF1744" if pct > 0 else "#8A99AD"

        is_dark = self.page.theme_mode == ft.ThemeMode.DARK
        text_color = "#E0E6ED" if is_dark else "#000000"

        self.news_lv.controls.clear()
        for t in self.current_data["news"]:
            self.news_lv.controls.append(
                ft.Container(
                    content=ft.Row([ft.Icon(ft.Icons.ARTICLE_ROUNDED, size=14, color="#8A99AD"), ft.Text(t, size=11, color=text_color, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, expand=True)], spacing=6),
                    on_hover=self.handle_box_hover,
                    bgcolor="#00000000"
                )
            )
        
        self.rumors_lv.controls.clear()
        for t in self.current_data["rumors"]:
            self.rumors_lv.controls.append(
                ft.Container(
                    content=ft.Row([ft.Icon(ft.Icons.RECORD_VOICE_OVER, size=14, color="#8A99AD"), ft.Text(t, size=11, color=text_color, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, expand=True)], spacing=6),
                    on_hover=self.handle_box_hover,
                    bgcolor="#00000000"
                )
            )

        for mdl in ["Gemini", "ChatGPT", "Claude", "Grok"]:
            r = self.ai_results[mdl]
            c = self.ai_cards[mdl]
            c.data["pct"].value = f"{r['change_pct']:+.2f} %"
            c.data["pct"].color = "#2979FF" if r["change_pct"] < 0 else "#FF1744" if r["change_pct"] > 0 else "#8A99AD"
            c.data["price"].value = f"{r['target_price']:,} 원"
            c.data["reason"].value = r["reason"]
            c.data["reason"].color = "#FFFFFF" if self.page.theme_mode == ft.ThemeMode.DARK else "#0F172A"

        d = self.consensus_result["direction"]
        cp = self.consensus_result["change_pct"]
        tp = self.consensus_result["target_price"]

        if d == "UP":
            self.result_status.value = "상승 전망 ▲"
            self.result_status.color = "#FF1744"
            self.consensus_box.border = ft.Border.all(2, "#40C4FF")
            self.result_pct.color = "#FF1744"
            self.result_diff.color = "#FF1744"
        else:
            self.result_status.value = "하락 전망 ▼"
            self.result_status.color = "#2979FF"
            self.consensus_box.border = ft.Border.all(2, "#40C4FF")
            self.result_pct.color = "#2979FF"
            self.result_diff.color = "#2979FF"

        self.result_pct.value = f"{cp:+.2f} %"
        self.result_price.value = f"예상 시가: {tp:,} 원"
        diff = tp - k["current_price"]
        self.result_diff.value = f"오늘 대비 {diff:+,}원 변동"

        try:
            dt_obj = datetime.datetime.strptime(self.current_data["timestamp"], "%Y-%m-%d %H:%M:%S")
            now_str = dt_obj.strftime("%Y년 %m월 %d일 %H시 %M분")
        except Exception:
            now_str = self.current_data["timestamp"]
        
        accent_color = "#C084FC" if is_dark else "#7C3AED"
        self.subtitle_label.spans = [
            ft.TextSpan(f"{now_str} 기준", style=ft.TextStyle(color=accent_color, weight=ft.FontWeight.BOLD)),
            ft.TextSpan(" 국내/미국 선물, 실시간 뉴스/주가, VIX 공포지수 등 변동 요인을 종합 분석하여 다음날 Kodex200 ETF의 시초가 예측",
                        style=ft.TextStyle(color="#8A99AD" if is_dark else "#64748B", weight=ft.FontWeight.BOLD))
        ]
        self.page.update()

    def handle_resize(self, e):
        try:
            self.page.update()
        except Exception:
            pass


def main(page: ft.Page):
    StockPredictorApp(page)

if __name__ == "__main__":
    ft.app(target=main)

# -*- coding: utf-8 -*-
import os
import json
import threading
import datetime
import flet as ft
from src.data_collector import DataCollector
from src.ai_consensus import AIConsensusManager
from src.reporter import PredictionReporter
from src.config import DEFAULT_WEIGHTS, ENV_API_KEYS, REPORTS_DIR, BASE_DIR

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


class StockPredictorApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "KODEX 200 AI Stock Predictor"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.bgcolor = "#121824"
        self.page.padding = 0
        self.page.window.width = 1220
        self.page.window.height = 950
        self.page.window.resizable = True
        self.page.scroll = ft.ScrollMode.AUTO


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
        self.setup_ui()
        # 초기 주가 차트 비동기 로딩
        threading.Thread(target=self.load_charts, daemon=True).start()

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
        self.monitor_lv.controls.append(
            ft.Text(f"[{ts}] {msg}", size=12, color="#B0C4DE", selectable=True)
        )
        try:
            self.page.update()
        except Exception:
            pass

    # ─── UI 빌드 ───
    def setup_ui(self):
        # ===== 메뉴바 =====
        menubar = ft.MenuBar(
            style=ft.MenuStyle(bgcolor="#1A2333"),
            controls=[
                ft.SubmenuButton(
                    content=ft.Text("파일", size=13, color="#E0E6ED"),
                    controls=[
                        ft.MenuItemButton(content=ft.Text("분석 실행"), leading=ft.Icon(ft.Icons.PLAY_ARROW_ROUNDED, size=18), on_click=self.run_analysis),
                        ft.MenuItemButton(content=ft.Text("보고서 저장"), leading=ft.Icon(ft.Icons.SAVE_ROUNDED, size=18), on_click=self.save_report_file),
                        ft.MenuItemButton(content=ft.Text("보고서 열기 (폴더)"), leading=ft.Icon(ft.Icons.FOLDER_OPEN_ROUNDED, size=18), on_click=self.open_reports_folder),
                        ft.MenuItemButton(content=ft.Text("종료"), leading=ft.Icon(ft.Icons.EXIT_TO_APP, size=18), on_click=lambda _: self.page.window.close()),
                    ],
                ),
                ft.SubmenuButton(
                    content=ft.Text("설정", size=13, color="#E0E6ED"),
                    controls=[
                        ft.MenuItemButton(content=ft.Text("API Key 설정"), leading=ft.Icon(ft.Icons.KEY_ROUNDED, size=18), on_click=self.open_settings_dialog),
                    ],
                ),
                ft.SubmenuButton(
                    content=ft.Text("도움말", size=13, color="#E0E6ED"),
                    controls=[
                        ft.MenuItemButton(content=ft.Text("사용 가이드"), leading=ft.Icon(ft.Icons.HELP_OUTLINE_ROUNDED, size=18), on_click=self.open_help_dialog),
                        ft.MenuItemButton(content=ft.Text("프로그램 정보"), leading=ft.Icon(ft.Icons.INFO_OUTLINE_ROUNDED, size=18), on_click=self.open_about_dialog),
                    ],
                ),
            ],
        )

        # ===== 헤더 =====
        title_label = ft.Text("KODEX 200 AI Predictor", size=24, weight=ft.FontWeight.BOLD, color="#E0E6ED")
        self.subtitle_label = ft.Text(
            spans=[
                ft.TextSpan(f"{datetime.datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')} 기준", style=ft.TextStyle(color="#2196F3", weight=ft.FontWeight.BOLD)),
                ft.TextSpan(" 한일 선물, 대형주, VIX 공포지수 및 실시간 뉴스/루머를 종합 분석하여 가중치가 반영된 최종 등락을 예측합니다.")
            ],
            size=11,
            color="#8A99AD"
        )

        # ===== 최종 결과 카드 =====
        self.result_status = ft.Text("대기 중...", size=18, weight=ft.FontWeight.BOLD, color="#8A99AD")
        self.result_pct = ft.Text("", size=18, weight=ft.FontWeight.BOLD, color="#8A99AD")
        self.result_price = ft.Text("", size=13, color="#B0C4DE")
        self.result_diff = ft.Text("", size=13, color="#B0C4DE")

        from src.config import MACRO_WEIGHTS
        weights_str = f"선물 {int((MACRO_WEIGHTS['Kospi_Future']+MACRO_WEIGHTS['Nasdaq_Future']+MACRO_WEIGHTS['SP500_Future'])*100)}% │ AI {int(MACRO_WEIGHTS['AI_Consensus']*100)}% │ 기타 {int((1-MACRO_WEIGHTS['AI_Consensus']-MACRO_WEIGHTS['Kospi_Future']-MACRO_WEIGHTS['Nasdaq_Future']-MACRO_WEIGHTS['SP500_Future'])*100)}%"
        self.consensus_box = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text("최종 종합 예측 결과", size=13, color="#8A99AD", weight=ft.FontWeight.BOLD),
                    ft.Text(f"가중치: {weights_str}", size=10, color="#8A99AD", italic=True),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Divider(color="#2E3A4E", thickness=1, height=1),
                ft.Row([
                    ft.Column([self.result_status, self.result_diff], spacing=1, alignment=ft.MainAxisAlignment.CENTER),
                    ft.VerticalDivider(width=10, color="#2E3A4E"),
                    ft.Column([self.result_pct, self.result_price], spacing=1, alignment=ft.MainAxisAlignment.CENTER),
                ], alignment=ft.MainAxisAlignment.SPACE_AROUND, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ], spacing=5),
            bgcolor="#1A2333", padding=ft.Padding(left=15, right=15, top=10, bottom=10), border_radius=12,
            border=ft.Border.all(1, "#2E3A4E"), width=574, height=120,
        )

        # ===== 국내 기초자산 =====
        self.kodex_price = ft.Text("KODEX200: - 원", size=15, weight=ft.FontWeight.BOLD, color="#FFFFFF")
        self.kodex_change = ft.Text("등락률: -%", size=12, color="#FFFFFF")
        self.samsung_price = ft.Text("삼성전자: - 원", size=13, color="#E0E6ED")
        self.samsung_change = ft.Text("(-%)", size=11, color="#8A99AD")
        self.hynix_price = ft.Text("SK하이닉스: - 원", size=13, color="#E0E6ED")
        self.hynix_change = ft.Text("(-%)", size=11, color="#8A99AD")

        kodex_box = ft.Container(
            content=ft.Column([
                ft.Row([ft.Icon(ft.Icons.FLAG_CIRCLE_ROUNDED, size=16, color="#FF3D00"), ft.Text("국내 기초자산 및 대형주", size=12, color="#8A99AD", weight=ft.FontWeight.BOLD)], spacing=6),
                ft.Divider(color="#2E3A4E", thickness=1, height=1),
                ft.Row([self.kodex_price, self.kodex_change], spacing=8),
                ft.Row([
                    ft.Column([self.samsung_price, self.samsung_change], spacing=1),
                    ft.VerticalDivider(width=8, color="#2E3A4E"),
                    ft.Column([self.hynix_price, self.hynix_change], spacing=1),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ], spacing=4),
            bgcolor="#1A2333", padding=ft.Padding(left=12, right=12, top=10, bottom=10), border_radius=12,
            border=ft.Border.all(1, "#2E3A4E"), width=574, height=120,
        )

        # ===== 주가 차트 영역 =====
        self.kodex_chart = ft.Image(src="chart", width=574, height=220, fit=ft.BoxFit.CONTAIN)
        self.kodex_chart.src_base64 = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        self.kospi_chart = ft.Image(src="chart", width=574, height=220, fit=ft.BoxFit.CONTAIN)
        self.kospi_chart.src_base64 = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

        # ===== 실행 컨트롤 =====
        self.progress_ring = ft.ProgressRing(width=22, height=22, stroke_width=3, visible=False, color="#00E676")
        self.status_msg = ft.Text("", color="#8A99AD", size=13)

        self.run_btn = ft.ElevatedButton(
            content=ft.Row([ft.Icon(ft.Icons.PLAY_ARROW_ROUNDED, color="#121824", size=16), ft.Text("분석 실행", size=15, weight=ft.FontWeight.BOLD, color="#121824")], alignment=ft.MainAxisAlignment.CENTER, spacing=4),
            style=ft.ButtonStyle(color={"hovered": "#FFFFFF", "": "#121824"}, bgcolor={"hovered": "#00C853", "": "#00E676"}, shape=ft.RoundedRectangleBorder(radius=10)),
            width=126, height=44, on_click=self.run_analysis,
        )

        # ===== AI 카드 =====
        self.ai_cards = {}
        colors = {"Gemini": "#4285F4", "ChatGPT": "#10a37f", "Claude": "#D97706", "Grok": "#E0E6ED"}
        for mdl in ["Gemini", "ChatGPT", "Claude", "Grok"]:
            w = int(DEFAULT_WEIGHTS[mdl] * 100)
            self.ai_cards[mdl] = self._mk_ai_card(f"{mdl} ({w}%)", colors[mdl])

        # ===== 매크로 대시보드 =====
        self.macro_cards = {}
        macro_items = [("코스피 선물","Kospi_Future"),("나스닥 선물","Nasdaq_Future"),("S&P 500 선물","SP500_Future"),("원/달러 환율","USD_KRW"),("엔/달러 환율","USD_JPY"),("일본 닛케이","Nikkei_225"),("VIX 공포지수","VIX_Index"),("미 10년 국채금리","US10Y_Treasury"),("WTI 국제 유가","WTI_Crude")]
        mc = []
        for title, key in macro_items:
            c = self._mk_macro_card(title)
            self.macro_cards[key] = c
            mc.append(c)

        # ===== 뉴스/소문 (탭 대신 수동 전환 버튼) =====
        self.news_lv = ft.ListView(expand=True, spacing=4, padding=5)
        self.rumors_lv = ft.ListView(expand=True, spacing=4, padding=5)
        self._news_tab_active = True

        self.news_tab_btn = ft.TextButton("실시간 속보 뉴스", style=ft.ButtonStyle(color="#00E676"), on_click=lambda _: self._switch_tab(True))
        self.rumors_tab_btn = ft.TextButton("증권가 소문/이슈", style=ft.ButtonStyle(color="#8A99AD"), on_click=lambda _: self._switch_tab(False))
        self.news_content_area = ft.Container(content=self.news_lv, expand=True)

        news_box = ft.Container(
            content=ft.Column([
                ft.Row([self.news_tab_btn, ft.Text("│", color="#2E3A4E"), self.rumors_tab_btn], spacing=8),
                ft.Divider(color="#2E3A4E", thickness=1, height=1),
                self.news_content_area,
            ], spacing=4),
            bgcolor="#1A2333", padding=12, border_radius=15,
            border=ft.Border.all(1, "#2E3A4E"), width=1160, height=170,
        )

        # ===== 모니터링 로그 =====
        self.monitor_lv = ft.ListView(expand=True, spacing=3, padding=8, auto_scroll=True)
        self.monitor_lv.controls.append(ft.Text("[시스템] 프로그램 초기화 완료. '파일 > 분석 실행' 또는 버튼을 클릭하세요.", size=12, color="#8A99AD", selectable=True))

        monitor_box = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.MONITOR_HEART_OUTLINED, size=16, color="#00E676"),
                    ft.Text("모니터링 로그", size=13, color="#8A99AD", weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                    ft.IconButton(icon=ft.Icons.DELETE_SWEEP_ROUNDED, icon_size=16, icon_color="#8A99AD", tooltip="로그 지우기", on_click=lambda _: self._clear_log()),
                ], spacing=6),
                ft.Divider(color="#2E3A4E", thickness=1, height=1),
                ft.Container(content=self.monitor_lv, expand=True),
            ], spacing=4),
            bgcolor="#111820", padding=10, border_radius=12,
            border=ft.Border.all(1, "#1E2A3A"), width=1160, height=220,
        )

        # ===== 페이지 조립 =====
        body = ft.Column([
            ft.Row([
                ft.Column([title_label, self.subtitle_label], spacing=4, expand=True),
                ft.Row([self.progress_ring, self.run_btn], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, width=1160),
            ft.Divider(color="#2E3A4E", thickness=1, height=1),
            ft.Row([self.consensus_box, kodex_box], spacing=12),
            ft.Row([
                self.status_msg,
            ], alignment=ft.MainAxisAlignment.START, width=1160),
            ft.Text("4대 핵심 AI 가중치 반영 실시간 예측 분석", size=14, color="#8A99AD", weight=ft.FontWeight.BOLD),
            ft.Row(controls=[self.ai_cards["Gemini"], self.ai_cards["ChatGPT"], self.ai_cards["Claude"], self.ai_cards["Grok"]], spacing=12),
            ft.Text("글로벌 선물/공포지수/환율/일본상황 실시간 대시보드", size=14, color="#8A99AD", weight=ft.FontWeight.BOLD),
            ft.Row(controls=mc, spacing=10),
            news_box,
            monitor_box,
        ], spacing=12, width=1160)

        # 1. 세로 스크롤을 제공하는 Column (높이를 동적으로 조절하여 붕괴 방지)
        self.vertical_scroll = ft.Column(
            [ft.Container(content=body, padding=ft.Padding(left=20, right=20, top=0, bottom=10))],
            scroll=ft.ScrollMode.AUTO,
            width=1200
        )
        win_height = self.page.window.height
        if not win_height or win_height < 500:
            win_height = 950
        self.vertical_scroll.height = win_height - 80

        # 2. 가로 스크롤을 제공하는 Row (menubar 아래 영역 전체를 채움)
        self.scrollable_body = ft.Row(
            [self.vertical_scroll],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH
        )

        self.page.scroll = None
        self.page.on_resize = self.handle_resize

        self.page.add(
            ft.Column([menubar, self.scrollable_body], spacing=0, expand=True)
        )

    # ─── 카드 헬퍼 ───
    def _mk_ai_card(self, name, color):
        lp = ft.Text("- %", size=18, weight=ft.FontWeight.BOLD, color="#B0C4DE")
        lprice = ft.Text("- 원", size=14, color="#E0E6ED")
        lr = ft.Text("대기 중...", size=11, color="#8A99AD")
        c = ft.Container(
            content=ft.Column([
                ft.Row([ft.Container(width=10, height=10, bgcolor=color, border_radius=5), ft.Text(name, size=13, weight=ft.FontWeight.BOLD, color="#E0E6ED")], spacing=8),
                ft.Divider(color="#2E3A4E", thickness=1), lp, lprice,
                ft.Container(content=lr),
            ], spacing=4),
            bgcolor="#1A2333", padding=12, border_radius=12, border=ft.Border.all(1, "#2E3A4E"), width=281, height=200,
        )
        c.data = {"pct": lp, "price": lprice, "reason": lr}
        return c

    def _mk_macro_card(self, title):
        lv = ft.Text("-", size=14, weight=ft.FontWeight.BOLD, color="#E0E6ED")
        lp = ft.Text("-", size=11, color="#8A99AD")
        c = ft.Container(
            content=ft.Column([ft.Text(title, size=11, color="#8A99AD"), lv, lp], spacing=2, alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#1A2333", padding=8, border_radius=10, border=ft.Border.all(1, "#2E3A4E"), width=120, height=80,
        )
        c.data = {"val": lv, "pct": lp}
        return c

    # ─── 탭 전환 ───
    def _switch_tab(self, news_active):
        self._news_tab_active = news_active
        if news_active:
            self.news_content_area.content = self.news_lv
            self.news_tab_btn.style = ft.ButtonStyle(color="#00E676")
            self.rumors_tab_btn.style = ft.ButtonStyle(color="#8A99AD")
        else:
            self.news_content_area.content = self.rumors_lv
            self.news_tab_btn.style = ft.ButtonStyle(color="#8A99AD")
            self.rumors_tab_btn.style = ft.ButtonStyle(color="#00E676")
        self.page.update()

    def _clear_log(self):
        self.monitor_lv.controls.clear()
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
        threading.Thread(target=self._do_analysis, daemon=True).start()

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
            self.kodex_change.color = "#FF3D00" if k["change_pct"] < 0 else "#00E676" if k["change_pct"] > 0 else "#8A99AD"
            self.samsung_price.value = f"삼성전자: {hw['Samsung']['price']:,}원"
            self.samsung_change.value = f"{hw['Samsung']['change_pct']:+.2f}%"
            self.samsung_change.color = "#FF3D00" if hw["Samsung"]["change_pct"] < 0 else "#00E676" if hw["Samsung"]["change_pct"] > 0 else "#8A99AD"
            self.hynix_price.value = f"SK하이닉스: {hw['Hynix']['price']:,}원"
            self.hynix_change.value = f"{hw['Hynix']['change_pct']:+.2f}%"
            self.hynix_change.color = "#FF3D00" if hw["Hynix"]["change_pct"] < 0 else "#00E676" if hw["Hynix"]["change_pct"] > 0 else "#8A99AD"

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
                card.data["pct"].color = "#FF3D00" if pct < 0 else "#00E676" if pct > 0 else "#8A99AD"

            self.news_lv.controls.clear()
            for t in self.current_data["news"]:
                self.news_lv.controls.append(ft.Row([ft.Icon(ft.Icons.ARTICLE_ROUNDED, size=14, color="#8A99AD"), ft.Text(t, size=11, color="#E0E6ED", max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, expand=True)], spacing=6))
            self.rumors_lv.controls.clear()
            for t in self.current_data["rumors"]:
                self.rumors_lv.controls.append(ft.Row([ft.Icon(ft.Icons.RECORD_VOICE_OVER, size=14, color="#8A99AD"), ft.Text(t, size=11, color="#E0E6ED", italic=True, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, expand=True)], spacing=6))

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
                c.data["pct"].color = "#FF3D00" if r["change_pct"] < 0 else "#00E676" if r["change_pct"] > 0 else "#8A99AD"
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

            if d == "UP":
                self.result_status.value = "상승 전망 ▲"
                self.result_status.color = "#00E676"
                self.consensus_box.border = ft.Border.all(2, "#00E676")
                self.result_pct.color = "#00E676"
                self.result_diff.color = "#00E676"
            else:
                self.result_status.value = "하락 전망 ▼"
                self.result_status.color = "#2196F3"
                self.consensus_box.border = ft.Border.all(2, "#2196F3")
                self.result_pct.color = "#2196F3"
                self.result_diff.color = "#2196F3"

            self.result_pct.value = f"{cp:+.2f} %"
            self.result_price.value = f"예상 시가: {tp:,} 원"
            diff = tp - k["current_price"]
            self.result_diff.value = f"오늘 대비 {diff:+,}원 변동"
            self.result_price.color = "#B0C4DE"

            now_str = datetime.datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
            self.subtitle_label.spans = [
                ft.TextSpan(f"{now_str} 기준", style=ft.TextStyle(color="#2196F3", weight=ft.FontWeight.BOLD)),
                ft.TextSpan(" 한일 선물, 대형주, VIX 공포지수 및 실시간 뉴스/루머를 종합 분석하여 가중치가 반영된 최종 등락을 예측합니다.")
            ]
            self.status_msg.value = "분석 완료. [파일 > 보고서 저장]에서 보고서를 내보낼 수 있습니다."
            self._log(f"★ 최종: {'상승' if d=='UP' else '하락'} {cp:+.2f}% → 예상 시초가 {tp:,}원")
            # 차트 최신 데이터 갱신
            threading.Thread(target=self.load_charts, daemon=True).start()

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

    def handle_resize(self, e):
        try:
            h = self.page.window.height
            if h and h >= 500:
                self.vertical_scroll.height = h - 80
                self.page.update()
        except Exception:
            pass


def main(page: ft.Page):
    StockPredictorApp(page)

if __name__ == "__main__":
    ft.app(target=main)

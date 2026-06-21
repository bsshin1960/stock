# -*- coding: utf-8 -*-
import os
import re
import json
import random
import traceback
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
from src.config import DEFAULT_WEIGHTS, BASE_DIR, MACRO_WEIGHTS

def _get_configured_macro_weights() -> dict:
    settings_file = BASE_DIR / "settings.json"
    if settings_file.exists():
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                custom_weights = saved.get("macro_weights")
                if custom_weights:
                    # 현재 유효한 키만 필터링 (구버전 키 제거)
                    valid_keys = set(MACRO_WEIGHTS.keys())
                    filtered = {k: float(v) for k, v in custom_weights.items() if k in valid_keys}
                    # 새 키에 대해 기본값 추가
                    for k, v in MACRO_WEIGHTS.items():
                        if k not in filtered:
                            filtered[k] = v
                    return filtered
        except Exception:
            pass
    return MACRO_WEIGHTS.copy()

def _get_configured_consensus_weights() -> dict:
    settings_file = BASE_DIR / "settings.json"
    if settings_file.exists():
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                custom_weights = saved.get("consensus_weights")
                if custom_weights:
                    return {k: float(v) for k, v in custom_weights.items()}
        except Exception:
            pass
    from src.config import CONSENSUS_WEIGHTS
    return CONSENSUS_WEIGHTS.copy()

def _get_configured_ai_weights() -> dict:
    settings_file = BASE_DIR / "settings.json"
    if settings_file.exists():
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                custom_weights = saved.get("ai_weights")
                if custom_weights:
                    return {k: float(v) for k, v in custom_weights.items()}
        except Exception:
            pass
    from src.config import DEFAULT_WEIGHTS
    return DEFAULT_WEIGHTS.copy()

def update_weights_from_feedback(entry: dict, actual_dir: str) -> dict:
    """
    history.json의 한 기록에 대해, 실제 예측 방향과 각 요인의 기여 방향을 비교하여
    Consensus 가중치, 매크로 지표 가중치, 그리고 개별 AI 가중치를 업데이트합니다.
    """
    # 1. Consensus 가중치 학습
    consensus_weights = entry.get("before_consensus_weights")
    if not consensus_weights:
        consensus_weights = _get_configured_consensus_weights()
        
    components = entry.get("components")
    if not components:
        return {}
        
    eta_c = 0.02
    updated_c = consensus_weights.copy()
    
    y = 1.0 if actual_dir == "UP" else -1.0 if actual_dir == "DOWN" else 0.0
    
    if y != 0.0:
        for comp, val in components.items():
            comp_dir_val = 1.0 if val >= 0 else -1.0
            if comp_dir_val == y:
                updated_c[comp] += eta_c
            else:
                updated_c[comp] -= eta_c
                
        for comp in updated_c:
            updated_c[comp] = max(0.01, updated_c[comp])
        total_c = sum(updated_c.values())
        for comp in updated_c:
            updated_c[comp] = round(updated_c[comp] / total_c, 4)
            
    # 2. 매크로 가중치 학습
    macro_weights = entry.get("before_macro_weights")
    if not macro_weights:
        macro_weights = _get_configured_macro_weights()
        
    macro_data = entry.get("macro_data")
    eta_m = 0.01
    updated_m = macro_weights.copy()
    
    if y != 0.0 and macro_data:
        for key, val in macro_data.items():
            w_i = updated_m.get(key, 0.0)
            c_i = val
            contribution = c_i * w_i * y
            
            if contribution > 0:
                if w_i > 0:
                    updated_m[key] += eta_m
                elif w_i < 0:
                    updated_m[key] -= eta_m
            elif contribution < 0:
                if w_i > 0:
                    updated_m[key] = max(0.0, w_i - eta_m)
                elif w_i < 0:
                    updated_m[key] = min(0.0, w_i + eta_m)
            else:
                if w_i == 0.0 and c_i != 0.0:
                    c_sgn = 1.0 if c_i > 0 else -1.0
                    if c_sgn == y:
                        updated_m[key] = 0.01 * y
                        
        total_abs = sum(abs(v) for v in updated_m.values())
        if total_abs > 0:
            for key in updated_m:
                updated_m[key] = round(updated_m[key] / total_abs, 4)
                
    # 현재 유효한 키만 필터링하여 반환
    valid_keys = set(MACRO_WEIGHTS.keys())
    updated_m = {k: v for k, v in updated_m.items() if k in valid_keys}
    # 누락된 유효 키 보완
    for k, v in MACRO_WEIGHTS.items():
        if k not in updated_m:
            updated_m[k] = v

    # 3. AI 모델 개별 가중치 학습
    ai_weights = entry.get("before_ai_weights")
    if not ai_weights:
        ai_weights = _get_configured_ai_weights()
        
    ai_predictions = entry.get("ai_predictions")
    eta_a = 0.02
    updated_a = ai_weights.copy()
    
    if y != 0.0 and ai_predictions:
        for mdl, pred in ai_predictions.items():
            pred_dir = pred.get("predicted_direction")
            pred_dir_val = 1.0 if pred_dir == "UP" else -1.0 if pred_dir == "DOWN" else 0.0
            if pred_dir_val == y:
                updated_a[mdl] += eta_a
            else:
                updated_a[mdl] -= eta_a
                
        for mdl in updated_a:
            updated_a[mdl] = max(0.01, updated_a[mdl])
        total_a = sum(updated_a.values())
        for mdl in updated_a:
            updated_a[mdl] = round(updated_a[mdl] / total_a, 4)

    return {
        "consensus_weights": updated_c,
        "macro_weights": updated_m,
        "ai_weights": updated_a
    }

def save_feedback_weights(new_weights: dict):
    settings_file = BASE_DIR / "settings.json"
    saved = {}
    if settings_file.exists():
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except Exception:
            pass
            
    saved["consensus_weights"] = new_weights.get("consensus_weights")
    saved["macro_weights"] = new_weights.get("macro_weights")
    saved["ai_weights"] = new_weights.get("ai_weights")
    
    try:
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Error] Failed to save feedback weights: {e}")

class AIConsensusManager:
    """다중 AI API를 활용하여 의견을 수집하고 최종 종합 결과를 산출하는 클래스"""

    def __init__(self, api_keys: dict = None):
        """
        api_keys: {
            "Gemini": "...",
            "ChatGPT": "...",
            "Claude": "...",
            "Grok": "..."
        }
        """
        self.api_keys = api_keys or {}
        
    def _create_prompt(self, data: dict) -> str:
        """수집된 다양한 금융 지표(신규 변수 및 일본 여건 포함)를 분석용 프롬프트로 변환"""
        k = data["kodex200"]
        hw = data["heavyweights"]
        m = data["macro"]
        news_str = "\n".join([f"- {title}" for title in data["news"]])
        rumors_str = "\n".join([f"- {title}" for title in data["rumors"]])
        
        # 모델별 가중치 스트링
        ai_weights = _get_configured_ai_weights()
        weights_str = ", ".join([f"{model}: {weight*100:.1f}%" for model, weight in ai_weights.items()])

        timestamp_str = data.get("timestamp", "")
        import datetime
        try:
            time_part = timestamp_str.split(" ")[1]
            hour = int(time_part.split(":")[0])
        except Exception:
            hour = datetime.datetime.now().hour
            
        if hour < 9:
            target_desc = "오늘(당일) 오전 9시 시초가(Open Price)"
            current_desc = "현재가"
            base_time_desc = "금일 현재가"
        else:
            target_desc = "다음 영업일 오전 9시 시초가(Open Price)"
            current_desc = "금일 오후 3시 현재가"
            base_time_desc = "금일 3시 현재가"

        prompt = f"""
당신은 국내 최정상급 금융공학 퀀트 애널리스트이자 글로벌 매크로 주식 전문가입니다.
제공된 시장 정보를 바탕으로 **KODEX 200 (코스피 200 지수 ETF)**의 **{target_desc}**가 **{current_desc}({k['current_price']:,}원)** 대비 어떻게 변화할지 분석하십시오.

[분석 기초 데이터]
1. KODEX 200 및 영향력 대형주 현황:
   - KODEX 200 {base_time_desc}: {k['current_price']:,} 원 (전일 대비 {k['change_pct']}% 변동)
   - 삼성전자: {hw['Samsung']['price']:,} 원 (전일 대비 {hw['Samsung']['change_pct']}% 변동)
   - SK하이닉스: {hw['Hynix']['price']:,} 원 (전일 대비 {hw['Hynix']['change_pct']}% 변동)
   - KODEX 200 기술적 지표: 
     * 5일 이동평균선: {k['sma5']:,} 원 │ 20일선: {k['sma20']:,} 원 │ 60일선: {k['sma60']:,} 원
     * RSI(14): {k['rsi14']} (과매수/과매도)
     * MACD: {k['macd']} (Signal: {k['macd_signal']}, Hist: {k['macd_hist']})
     * 볼린저 밴드: 상한선 {k['bb_upper']:,} 원 │ 하한선 {k['bb_lower']:,} 원

2. 선물 시장 및 글로벌 거시 경제, 환율, 주요 주가지수, 원자재 및 암호화폐:
   - 코스피 200 선물: {m['Kospi_Future']['value']:,} ({m['Kospi_Future']['change_pct']}%)
   - 나스닥 100 선물: {m['Nasdaq_Future']['value']:,} ({m['Nasdaq_Future']['change_pct']}%)
   - S&P 500 선물: {m['SP500_Future']['value']:,} ({m['SP500_Future']['change_pct']}%)
   - 원/달러 환율: {m['USD_KRW']['value']:,} 원 ({m['USD_KRW']['change_pct']}%)
   - 달러/엔 환율: {m['USD_JPY']['value']:,} 엔 ({m['USD_JPY']['change_pct']}%)
   - 미국 소비자 물가지수(CPI): {m['US_CPI']['value']:,} ({m['US_CPI']['change_pct']}%)
   - CBOE VIX (공포 지수): {m['VIX_Index']['value']:,} ({m['VIX_Index']['change_pct']}%)
   - 미국 10년물 국채 금리: {m['US10Y_Treasury']['value']}% ({m['US10Y_Treasury']['change_pct']}%)
   - WTI 원유 선물: {m['WTI_Crude']['value']}$ ({m['WTI_Crude']['change_pct']}%)
   - 금 선물: {m['Gold_Future']['value']:,}$ ({m['Gold_Future']['change_pct']}%)
   - 한국은행 기준금리 (국내 금리): {m['KR_Rate']['value']}% ({m['KR_Rate']['change_pct']}%)
   - 국고채 3년 금리 (국내 채권금리): {m['KR_Bond']['value']}% ({m['KR_Bond']['change_pct']}%)
   - 필라델피아 반도체 지수: {m['SOX_Index']['value']:,} ({m['SOX_Index']['change_pct']}%)
   - US 달러 인덱스: {m['Dollar_Index']['value']:,} ({m['Dollar_Index']['change_pct']}%)
   - 일본 닛케이 225 지수: {m['Nikkei_225']['value']:,} ({m['Nikkei_225']['change_pct']}%)
   - 상해 종합 지수: {m['Shanghai_Composite']['value']:,} ({m['Shanghai_Composite']['change_pct']}%)
   - MSCI 한국 ETF 종가: {m['MSCI_Korea']['value']:,} ({m['MSCI_Korea']['change_pct']}%)
   - 상위종목공매도 (KODEX200): {m['Short_Selling']['value']:,} ({m['Short_Selling']['change_pct']}%)
   - 유명인사 발언 (연준의장/트럼프/이재명 등): {m['Famous_Remarks']['value']:,} ({m['Famous_Remarks']['change_pct']}%)
   - 비트코인 (BTC/USD): {m['Bitcoin']['value']:,}$ ({m['Bitcoin']['change_pct']}%)
   - 미국 기준금리 (미국 금리) (0%): {m['US_Rate']['value']}% ({m['US_Rate']['change_pct']}%)
   - 검토중(0%): {m['EUR_USD']['value']:,} ({m['EUR_USD']['change_pct']}%)

3. 당일 경제 뉴스 헤드라인:
{news_str}

4. 신뢰성 있는 증권가 소문(루머) 및 풍문:
{rumors_str}

5. 최종 합산 모델별 가중치 정보 (참고용):
- {weights_str}

[요구사항]
위 지표들의 상관관계(예: 코스피/나스닥 야간 선물 상승 시 시초가 갭상승, 대형주 강세 시 KODEX 200 상승 압력, 공포지수 VIX 급등 시 심리 위축 및 시가 하락 출발 압력, 달러/엔 환율에 따른 엔화 가치 변동 및 일본 닛케이 증시 흐름이 한국 수출 경쟁력에 미치는 영향, 환율 급등 시 외국인 매도세, 증권가 풍문 루머에 의한 특정 테마/섹터 심리 등)를 심층 분석하십시오.
그리고 반드시 최종 결론을 아래 JSON 형식으로만 응답해 주십시오. 텍스트 설명이나 마크다운 백틱(```json) 없이 순수 JSON 객체만 리턴해야 합니다.

{{
  "direction": "UP" 또는 "DOWN",  // 상승 예상 시 UP, 하락 예상 시 DOWN
  "change_pct": 0.85,             // 소수점 둘째자리까지의 예상 등락률 (예: 0.85 또는 -0.45)
  "target_price": 328100,         // 예상되는 다음 영업일 시초가 금액 (정수형 원화 단위)
  "reason": "예측의 근거를 등락 결정 비중(가중치)이 높은 요인 순서대로 머리글을 붙여 각각 짧은 한 줄 문장으로 요약하여 반드시 5개 이상의 불릿 포인트로 작성하십시오. 각 줄은 반드시 검은색 원형 불릿 기호(•)로 시작해야 합니다. (예: '• [코스피 선물] ...\\n• [나스닥 선물] ...\\n• [S&P500 선물] ...\\n• [원/달러 환율] ...\\n• [삼성전자] ...') 문장이 너무 길어지지 않게 간결하고 직관적으로 작성해 주십시오."
}}
"""
        return prompt

    def _get_mock_prediction(self, model_name: str, data: dict) -> dict:
        """API Key가 없거나 호출 실패 시 고품질 시뮬레이션 데이터 제공 (일본 상황 추가 연동)"""
        k = data["kodex200"]
        hw = data["heavyweights"]
        m = data["macro"]
        rumors = data["rumors"]
        
        current_price = k["current_price"]
        
        # 모델별 독립 분석 기준을 모방하여 가중치 배분 및 시뮬레이션 지수 계산
        score = 0.0
        
        # 루머/이슈에 따른 가중 편차 (특정 단어 존재 시 가산/감산)
        rumor_sentiment = 0.0
        for rum in rumors:
            if "공급" in rum or "호조" in rum or "유입" in rum or "비둘기" in rum:
                rumor_sentiment += 0.2
            if "우려" in rum or "매파" in rum or "감소" in rum or "차질" in rum:
                rumor_sentiment -= 0.2
        
        # 모델별 차별화된 예측 공식 적용 (스케일링 하향 및 모델별 오프셋 조정)
        ta1_pct = m.get("Technical_Analysis1", {}).get("change_pct", 0.0)
        ta2_pct = m.get("Technical_Analysis2", {}).get("change_pct", 0.0)

        if model_name == "Gemini":
            # Gemini: 국내 대형주 흐름 + 코스피 선물 + 기술적 지표(RSI) 위주 분석
            score += hw["Samsung"]["change_pct"] * 1.2
            score += hw["Hynix"]["change_pct"] * 0.6
            score += m["Kospi_Future"]["change_pct"] * 1.0
            rsi = k["rsi14"] or 50
            if rsi < 30:
                score += 0.5
            elif rsi > 70:
                score -= 0.5
            score += rumor_sentiment * 0.4
            score += ta1_pct * 0.4  # 최근주가분석 변동폭 반영
            predicted_pct = score * 0.10 + 0.05
            
        elif model_name == "ChatGPT":
            # ChatGPT: 글로벌/국내 선물 시장 상관성 위주 분석
            score += m["Kospi_Future"]["change_pct"] * 1.4
            score += m["Nasdaq_Future"]["change_pct"] * 1.0
            score += m["SP500_Future"]["change_pct"] * 0.6
            score -= (m["VIX_Index"]["value"] - 15.0) * 0.03
            score += rumor_sentiment * 0.6
            score += ta2_pct * 0.5  # 기술적 보조지표 종합 시그널 반영
            predicted_pct = score * 0.08 - 0.05
            
        elif model_name == "Claude":
            # Claude: 거시 경제 지표 및 미국 소비자물가지수(CPI)(환율, 미국소비자물가지수, 10년물 국채) 위주 분석
            score -= m["US_CPI"]["change_pct"] * 1.0
            score -= m["USD_KRW"]["change_pct"] * 1.2
            score -= m["USD_JPY"]["change_pct"] * 0.6
            score -= m["US10Y_Treasury"]["change_pct"] * 0.3
            score += hw["Samsung"]["change_pct"] * 0.5
            score += rumor_sentiment * 0.5
            score += (ta1_pct + ta2_pct) * 0.2  # 복합 기술 지표 반영
            predicted_pct = score * 0.07 + 0.10
            
        else: # Grok
            # Grok: 뉴스 및 풍문 감성 지표 + 국제 유가 + 공포 지수(VIX) 위주 분석
            score -= (m["VIX_Index"]["value"] - 15.0) * 0.08
            score += m["WTI_Crude"]["change_pct"] * 0.6
            score += m["Kospi_Future"]["change_pct"] * 0.6
            score -= m["USD_KRW"]["change_pct"] * 0.4
            score += rumor_sentiment * 1.0
            score += ta1_pct * 0.3  # 최근주가 변동성 반영
            predicted_pct = score * 0.09 - 0.10

        # 모델별 고유 편차 추가 (동적 변동성 확보 및 Seed 고정)
        random.seed(hash(model_name + str(current_price)))
        model_variance = random.uniform(-0.40, 0.40)
        
        predicted_pct = round(predicted_pct + model_variance, 2)
        
        # 모델별 클리핑 한계 조정을 통해 극단적인 시장 상황에서도 수치가 동일하게 겹치는 현상 방지
        if model_name == "Gemini":
            clip_min, clip_max = -3.00, 3.00
        elif model_name == "ChatGPT":
            clip_min, clip_max = -2.95, 2.95
        elif model_name == "Claude":
            clip_min, clip_max = -2.90, 2.90
        else: # Grok
            clip_min, clip_max = -2.85, 2.85
            
        predicted_pct = max(clip_min, min(clip_max, predicted_pct))
        
        direction = "UP" if predicted_pct >= 0 else "DOWN"
        target_price = int(current_price * (1 + predicted_pct / 100))
        
        # 풍문/소문을 분석 이유에 임베딩
        selected_rumor = rumors[0] if rumors else "증권가 풍문 요인 미비"
        clean_rumor = selected_rumor.replace("[단독 루머] ", "").replace("[소문] ", "").replace("[찌라시] ", "").replace("[반도체설] ", "").replace("[수급 소문] ", "")
        
        reasons = {
            "Gemini": f"• [코스피 선물] 야간 코스피 선물지수 상승에 따른 시초가 갭상승 압력 우세\n• [나스닥 선물] 미국 나스닥 선물 상승으로 글로벌 위험자산 선호 심리 동조화\n• [삼성전자] 삼성전자 {hw['Samsung']['change_pct']}% 반등 흐름에 의한 지수 하방 지지력 확보\n• [미국 소비자 물가지수] 미국 소비자 물가지수(CPI) 변동률 {m['US_CPI']['change_pct']}%이 글로벌 금리 및 인플레이션 센티먼트에 미치는 영향\n• [기술적 분석] 최근 5일 이동평균선 대비 이격률({ta1_pct:+.2f}%) 추세를 감안한 단기 수급 지지선 확인",
            "ChatGPT": f"• [코스피 선물] 국내 야간 선물의 우상향 기조로 지수 강보합 출발 우세\n• [나스닥 선물] 미국 지수 선물의 강세로 장 초반 기술주 매수 동조화\n• [삼성전자] 삼성전자 반도체 공급 계약 소식에 따른 장중 심리 개선\n• [엔화 가치] 엔화 약세 여파(환율 변동률: {m['USD_JPY']['change_pct']}%)에 따른 IT/자동차 대형 수출주 센티먼트 변화\n• [기술적 분석] RSI 및 MACD 보조지표 종합 분석 시그널 점수({ta2_pct:+.1f}점) 기반의 장초반 변동 예상",
            "Claude": f"• [코스피 선물] 야간 코스피 선물 지수의 등락을 통한 장초반 방향성 결정\n• [삼성전자] 반도체 대형주 중심의 외국인 장초반 순매수 기대감 반영\n• [엔/달러 환율] 엔/달러 환율({m['USD_JPY']['value']}엔) 변동에 따른 아시아 대형 제조사 수급 자극\n• [원/달러 환율] 원/달러 환율 변동성({m['USD_KRW']['change_pct']}%)에 의한 외인 매수 압력 가중\n• [기술적 분석] 최근 주가 이격도({ta1_pct:+.2f}%) 및 RSI/MACD 종합 보조 분석을 연계한 매수 강도 판단",
            "Grok": f"• [코스피 선물] 야간 코스피 선물 지수의 견조한 지지로 지수 하방 경직성 확보\n• [나스닥 선물] 나스닥 및 S&P 500 선물의 단기 추세 추종에 따른 동조 상승\n• [공포지수] VIX 공포지수({m['VIX_Index']['value']}) 하락세 전환에 따른 시장 안도 랠리 심리\n• [원/달러 환율] 원/달러 환율 안정세 진입 시 외국인 장중 현선물 매수 유입 가능성\n• [기술적 분석] KODEX 200 최근주가 괴리 지표({ta1_pct:+.2f}%)에 근거한 기술적 하방 지지 강도 분석"
        }
        
        return {
            "direction": direction,
            "change_pct": predicted_pct,
            "target_price": target_price,
            "reason": reasons.get(model_name, "시장 지표 종합 분석에 따른 예측치입니다.")
        }

    def _parse_ai_response(self, text: str, model_name: str, data: dict) -> dict:
        """AI의 응답 텍스트에서 JSON 데이터를 안전하게 파싱"""
        try:
            # 마크다운 백틱 제거
            clean_text = text.strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()
            
            # JSON만 찾아내기 위한 중괄호 매칭
            match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if match:
                clean_text = match.group(0)
                
            parsed = json.loads(clean_text)
            
            # 필수 키 확인 및 가공
            direction = parsed.get("direction", "UP").upper()
            change_pct = float(parsed.get("change_pct", 0.0))
            target_price = int(parsed.get("target_price", data["kodex200"]["current_price"]))
            reason = parsed.get("reason", "분석이 정상 완료되었습니다.")
            
            return {
                "direction": direction,
                "change_pct": change_pct,
                "target_price": target_price,
                "reason": reason
            }
        except Exception as e:
            print(f"[Warning] {model_name} 응답 파싱 실패 ({e}). 시뮬레이션 데이터를 제공합니다.")
            return self._get_mock_prediction(model_name, data)

    # --- 실 API 호출 함수들 ---
    
    def _call_gemini(self, prompt: str, key: str, data: dict) -> dict:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            return self._parse_ai_response(response.text, "Gemini", data)
        except Exception as e:
            print(f"[Error] Gemini API 호출 실패: {e}")
            return self._get_mock_prediction("Gemini", data)

    def _call_chatgpt(self, prompt: str, key: str, data: dict) -> dict:
        try:
            client = OpenAI(api_key=key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            return self._parse_ai_response(response.choices[0].message.content, "ChatGPT", data)
        except Exception as e:
            print(f"[Error] ChatGPT API 호출 실패: {e}")
            return self._get_mock_prediction("ChatGPT", data)

    def _call_claude(self, prompt: str, key: str, data: dict) -> dict:
        try:
            client = Anthropic(api_key=key)
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            # Claude 3는 text block 리스트를 반환함
            return self._parse_ai_response(response.content[0].text, "Claude", data)
        except Exception as e:
            print(f"[Error] Claude API 호출 실패: {e}")
            return self._get_mock_prediction("Claude", data)

    def _call_grok(self, prompt: str, key: str, data: dict) -> dict:
        try:
            # Grok은 xAI API 엔드포인트 사용 (OpenAI SDK 호환 가능)
            client = OpenAI(
                api_key=key,
                base_url="https://api.x.ai/v1"
            )
            response = client.chat.completions.create(
                model="grok-2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return self._parse_ai_response(response.choices[0].message.content, "Grok", data)
        except Exception as e:
            print(f"[Error] Grok API 호출 실패: {e}")
            return self._get_mock_prediction("Grok", data)

    def analyze_all_models(self, data: dict) -> dict:
        """모든 AI 모델의 의견을 개별 수집"""
        prompt = self._create_prompt(data)
        results = {}

        # 1. Gemini
        gemini_key = self.api_keys.get("Gemini")
        if gemini_key:
            print("[System] Gemini 실시간 분석 진행 중...")
            results["Gemini"] = self._call_gemini(prompt, gemini_key, data)
        else:
            print("[System] Gemini API Key 없음 (시뮬레이션 모드)")
            results["Gemini"] = self._get_mock_prediction("Gemini", data)

        # 2. ChatGPT
        chatgpt_key = self.api_keys.get("ChatGPT")
        if chatgpt_key:
            print("[System] ChatGPT 실시간 분석 진행 중...")
            results["ChatGPT"] = self._call_chatgpt(prompt, chatgpt_key, data)
        else:
            print("[System] ChatGPT API Key 없음 (시뮬레이션 모드)")
            results["ChatGPT"] = self._get_mock_prediction("ChatGPT", data)

        # 3. Claude
        claude_key = self.api_keys.get("Claude")
        if claude_key:
            print("[System] Claude 실시간 분석 진행 중...")
            results["Claude"] = self._call_claude(prompt, claude_key, data)
        else:
            print("[System] Claude API Key 없음 (시뮬레이션 모드)")
            results["Claude"] = self._get_mock_prediction("Claude", data)

        # 4. Grok
        grok_key = self.api_keys.get("Grok")
        if grok_key:
            print("[System] Grok 실시간 분석 진행 중...")
            results["Grok"] = self._call_grok(prompt, grok_key, data)
        else:
            print("[System] Grok API Key 없음 (시뮬레이션 모드)")
            results["Grok"] = self._get_mock_prediction("Grok", data)

        return results

    def calculate_consensus(self, data: dict, ai_results: dict, weights: dict = None) -> dict:
        """AI 개별 의견들과 매크로 지표들을 가중 결합하여 최종 종합 결과를 산출"""
        from src.config import MACRO_WEIGHTS, CONSENSUS_WEIGHTS
        
        if not weights:
            weights = _get_configured_ai_weights()
            
        current_price = data["kodex200"]["current_price"]
        news_list = data.get("news", [])
        rumors_list = data.get("rumors", [])
        
        # 1단계: AI 가중 합산 (전체 중 50% 반영)
        weighted_ai_change_pct = 0.0
        total_ai_weight = 0.0
        for model, res in ai_results.items():
            w = weights.get(model, 0.25)
            change = res["change_pct"]
            weighted_ai_change_pct += change * w
            total_ai_weight += w
            
        if total_ai_weight > 0:
            ai_consensus_change_pct = weighted_ai_change_pct / total_ai_weight
        else:
            ai_consensus_change_pct = 0.0
            
        # 2단계: 글로벌 매크로 지표 가중 합산 (전체 중 30% 반영)
        m = data["macro"]
        
        # 각 매크로 지표의 실질 변동률 (MACRO_WEIGHTS에 정의된 키만 안전하게 포함)
        factors = {}
        for key in MACRO_WEIGHTS.keys():
            if key in m:
                factors[key] = m[key].get("change_pct", 0.0)
            else:
                factors[key] = 0.0
        
        # 설정된 매크로 가중치 로드
        macro_weights = _get_configured_macro_weights()
        
        # 상황에 따른 지표 가중치 자동 보정 (동적 가중치 스케일링)
        # 예: 뉴스나 루머에 금리 인상/인하 등 금리 관련 중요 소식이 감지되면 국내 금리 및 채권금리 가중치를 대폭 늘림
        all_text_pool = " ".join(news_list + rumors_list)
        rate_scaling = 1.0
        rate_keywords = ["금리", "기준금리", "금리인상", "금리인하", "한국은행 금리", "채권금리", "인상 예정", "인하 예정", "금리 동결", "통화정책"]
        has_rate_issue = any(kw in all_text_pool for kw in rate_keywords)
        
        if has_rate_issue:
            rate_scaling = 2.0  # 금리 이슈 우려 시 국내 금리/채권 가중치 2배로 자동 강화!
            print(f"[System] 뉴스/풍문 분석 결과 금리 이슈 감지: 국내 금리/채권 가중치 자동 강화 ({rate_scaling}x)")
        
        macro_weighted_sum = 0.0
        abs_macro_weight_sum = 0.0
        
        for key, val in factors.items():
            w = macro_weights.get(key, 0.0)
            if key in ["KR_Rate", "KR_Bond"] and has_rate_issue:
                w *= rate_scaling
            macro_weighted_sum += val * w
            abs_macro_weight_sum += abs(w)
            
        if abs_macro_weight_sum > 0:
            macro_change_pct = macro_weighted_sum / abs_macro_weight_sum
        else:
            macro_change_pct = 0.0

        # 3단계: 실시간 속보 뉴스 감성 분석 (전체 중 15% 반영)
        news_list = data.get("news", [])
        news_pos_keywords = ["상승", "호재", "급등", "개선", "기대", "돌파", "반등", "활황", "출발", "뛰기", "유입", "안정", "매수세", "순매수"]
        news_neg_keywords = ["하락", "악재", "급락", "악화", "우려", "위험", "위축", "약세", "피눈물", "감소", "매도세", "순매도", "리스크"]
        
        news_pos_cnt = 0
        news_neg_cnt = 0
        for text in news_list:
            for w in news_pos_keywords:
                if w in text:
                    news_pos_cnt += 1
            for w in news_neg_keywords:
                if w in text:
                    news_neg_cnt += 1
                    
        if news_pos_cnt + news_neg_cnt > 0:
            news_sentiment = (news_pos_cnt - news_neg_cnt) / (news_pos_cnt + news_neg_cnt)
        else:
            news_sentiment = 0.0
        # 뉴스 감성 지수의 최대 변동 범위를 -1.5% ~ +1.5%로 설정
        news_change_pct = news_sentiment * 1.5
        
        # 4단계: 증권가 소문/이슈 감성 분석 (전체 중 5% 반영)
        rumors_list = data.get("rumors", [])
        rumors_pos_keywords = ["상승", "호재", "급등", "개선", "기대", "돌파", "반등", "활황", "수혜", "유입", "안정", "단독", "비둘기", "공급"]
        rumors_neg_keywords = ["하락", "악재", "급락", "악화", "우려", "위험", "위축", "약세", "피눈물", "매파", "유출", "조정", "악재"]
        
        rumor_pos_cnt = 0
        rumor_neg_cnt = 0
        for text in rumors_list:
            for w in rumors_pos_keywords:
                if w in text:
                    rumor_pos_cnt += 1
            for w in rumors_neg_keywords:
                if w in text:
                    rumor_neg_cnt += 1
                    
        if rumor_pos_cnt + rumor_neg_cnt > 0:
            rumor_sentiment = (rumor_pos_cnt - rumor_neg_cnt) / (rumor_pos_cnt + rumor_neg_cnt)
        else:
            rumor_sentiment = 0.0
        # 소문 감성 지수의 최대 변동 범위를 -1.0% ~ +1.0%로 설정
        rumor_change_pct = rumor_sentiment * 1.0

        # 5단계: 최종 종합 가중 합산 (AI, 매크로, 뉴스, 소문 가중치 로드)
        consensus_weights = _get_configured_consensus_weights()
        w_ai = consensus_weights.get("AI_Consensus", 0.50)
        w_macro = consensus_weights.get("Macro_Dashboard", 0.30)
        w_news = consensus_weights.get("News_Consensus", 0.15)
        w_rumor = consensus_weights.get("Rumor_Consensus", 0.05)

        consensus_change_pct = (ai_consensus_change_pct * w_ai) + (macro_change_pct * w_macro) + (news_change_pct * w_news) + (rumor_change_pct * w_rumor)
        consensus_change_pct = round(consensus_change_pct, 2)
            
        # 등락 한도 제한 (-3.0% ~ +3.0%)
        consensus_change_pct = max(-3.0, min(3.0, consensus_change_pct))
            
        # 예상 가격 계산 (오늘 3시 가격 기준)
        consensus_target_price = int(current_price * (1 + consensus_change_pct / 100))
        consensus_direction = "UP" if consensus_change_pct >= 0 else "DOWN"
        
        # 합의 종합 분석 리포트 생성
        reasons_summary = "   │   ".join([f"{model}: {res['change_pct']}%" for model, res in ai_results.items()])
        
        consensus_weights_desc = f"AI {w_ai*100:.1f}% │ 매크로 {w_macro*100:.1f}% │ 뉴스 {w_news*100:.1f}% │ 소문 {w_rumor*100:.1f}%"
        
        timestamp_str = data.get("timestamp", "")
        import datetime
        try:
            time_part = timestamp_str.split(" ")[1]
            hour = int(time_part.split(":")[0])
        except Exception:
            hour = datetime.datetime.now().hour
            
        target_day_desc = "오늘(당일) 오전 9시" if hour < 9 else "내일"

        consensus_reason = (
            f"4대 AI 분석 의견({w_ai*100:.1f}%), 글로벌 매크로 지표({w_macro*100:.1f}%), 실시간 속보 뉴스({w_news*100:.1f}%), 증권가 소문/이슈({w_rumor*100:.1f}%)를 종합 가중 분석한 결론입니다.\n"
            f"AI 예측 변동률({ai_consensus_change_pct:+.2f}%), 글로벌 매크로 평균 변동률({macro_change_pct:+.2f}%), "
            f"실시간 뉴스 분석({news_change_pct:+.2f}%), 소문 감성 지표({rumor_change_pct:+.2f}%)가 반영되었으며,\n"
            f"{target_day_desc} KODEX 200 시초가는 금일 대비 {consensus_change_pct:+.2f}% 변동한 {consensus_target_price:,}원 부근 형성이 유력합니다."
        )
        if has_rate_issue:
            consensus_reason += f"\n(※ 뉴스/풍문 금리 이슈 분석에 따라 국내 금리 및 채권 가중치가 {rate_scaling}배 자동 상향 적용되었습니다.)"
        
        return {
            "direction": consensus_direction,
            "change_pct": consensus_change_pct,
            "target_price": consensus_target_price,
            "reason": consensus_reason,
            "details": reasons_summary,
            "weights": weights,
            "components": {
                "AI_Consensus": ai_consensus_change_pct,
                "Macro_Dashboard": macro_change_pct,
                "News_Consensus": news_change_pct,
                "Rumor_Consensus": rumor_change_pct
            }
        }

if __name__ == "__main__":
    # 목 데이터 테스트 (모든 필수 키 포함)
    test_data = {
        "kodex200": {
            "current_price": 325400,
            "change_pct": -0.45,
            "sma5": 324500, "sma20": 323000, "sma60": 321500,
            "rsi14": 54.2, "macd": 120, "macd_signal": 100, "macd_hist": 20,
            "bb_upper": 329000, "bb_lower": 318000
        },
        "heavyweights": {
            "Samsung": {"price": 75200, "change_pct": 0.27},
            "Hynix": {"price": 182400, "change_pct": -1.15}
        },
        "macro": {
            "Kospi_Future": {"value": 324.50, "change_pct": -0.35},
            "Nasdaq_Future": {"value": 18230.5, "change_pct": 0.45},
            "SP500_Future": {"value": 5210.2, "change_pct": 0.32},
            "USD_KRW": {"value": 1365.20, "change_pct": -0.15},
            "USD_JPY": {"value": 156.40, "change_pct": 0.08},
            "Gold_Future": {"value": 2350.20, "change_pct": 0.35},
            "US10Y_Treasury": {"value": 4.432, "change_pct": -0.85},
            "WTI_Crude": {"value": 78.45, "change_pct": 0.52},
            "VIX_Index": {"value": 13.45, "change_pct": 1.25},
            "KR_Rate": {"value": 3.50, "change_pct": 0.0},
            "KR_Bond": {"value": 3.20, "change_pct": -0.15},
            "SOX_Index": {"value": 4920.50, "change_pct": 0.55},
            "Dollar_Index": {"value": 104.50, "change_pct": 0.12},
            "US_CPI": {"value": 314.02, "change_pct": 0.31},
            "Technical_Analysis1": {"value": 0.0, "change_pct": 0.0},
            "Technical_Analysis2": {"value": 0.0, "change_pct": 0.0},
            "Nikkei_225": {"value": 38720.50, "change_pct": -0.42},
            "Shanghai_Composite": {"value": 3110.25, "change_pct": 0.15},
            "MSCI_Korea": {"value": 62.45, "change_pct": 0.42},
            "Short_Selling": {"value": 1452.40, "change_pct": -0.85},
            "Famous_Remarks": {"value": 52.34, "change_pct": 0.45},
            "Bitcoin": {"value": 66250.00, "change_pct": 1.45},
            "US_Rate": {"value": 5.25, "change_pct": 0.0},
            "EUR_USD": {"value": 1.085, "change_pct": 0.05}
        },
        "news": ["미국 금리인하 기대감 솔솔", "반도체 칩 업황 대활황 예고"],
        "rumors": ["[단독 루머] HBM4 공급처 다변화에 따른 국내 대형 제조사의 납품 시기 2개월 앞당겨진다는 설"]
    }
    
    manager = AIConsensusManager()
    res = manager.analyze_all_models(test_data)
    con = manager.calculate_consensus(test_data, res)
    import pprint
    pprint.pprint(res)
    print("--- CONSENSUS ---")
    pprint.pprint(con)


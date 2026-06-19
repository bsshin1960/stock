# -*- coding: utf-8 -*-
import os
import re
import json
import random
import datetime
import traceback
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
from src.config import DEFAULT_WEIGHTS, BASE_DIR, MACRO_WEIGHTS, BASE_MACRO_WEIGHTS

def _get_configured_macro_weights(timestamp: str = None, macro_data: dict = None) -> dict:
    settings_file = BASE_DIR / "settings.json"
    weights = MACRO_WEIGHTS.copy()
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
                    weights = filtered
        except Exception:
            pass

    # 모든 가중치 초기값이 0%이고 실시간 분석(macro_data가 존재)이 진행되는 경우, 동적 연산을 위해 기본 베이스라인 가중치 적용
    if macro_data and all(v == 0.0 for v in weights.values()):
        weights = BASE_MACRO_WEIGHTS.copy()

    # 시간대에 따른 동적 가중치 조정 (나스닥 선물 vs 나스닥 주가)
    dt = None
    if timestamp:
        try:
            dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    if dt is None:
        dt = datetime.datetime.now()

    # US DST 및 개장 여부 판단
    year = dt.year
    dst_start = datetime.datetime(year, 3, 8)
    while dst_start.weekday() != 6:
        dst_start += datetime.timedelta(days=1)
    dst_end = datetime.datetime(year, 11, 1)
    while dst_end.weekday() != 6:
        dst_end += datetime.timedelta(days=1)
    is_dst = (dst_start <= dt < dst_end)
    
    open_hour, open_minute = (22, 30) if is_dst else (23, 30)
    curr_time = dt.time()
    open_time = datetime.time(open_hour, open_minute)
    morning_limit = datetime.time(9, 0)
    
    is_active = (curr_time >= open_time or curr_time < morning_limit)

    total_nasdaq_w = weights.get("Nasdaq_Future", 0.15) + weights.get("NASDAQ", 0.0)
    if is_active:
        weights["NASDAQ"] = round(total_nasdaq_w, 4)
        weights["Nasdaq_Future"] = 0.0
    else:
        weights["Nasdaq_Future"] = round(total_nasdaq_w, 4)
        weights["NASDAQ"] = 0.0

    # 변동율 크기에 따른 가중치 동적 스케일링 및 정규화
    scaled_weights = {}
    sum_abs = 0.0
    for key, w in weights.items():
        if w == 0.0:
            scaled_weights[key] = 0.0
            continue
            
        change_pct = 0.0
        if macro_data:
            val = macro_data.get(key)
            if isinstance(val, dict):
                change_pct = val.get("change_pct", 0.0)
            elif isinstance(val, (int, float)):
                change_pct = val
                
        # 스케일 공식: w * (1.0 + alpha * abs(change_pct))
        alpha = 1.0
        scale = 1.0 + alpha * abs(change_pct)
        w_new = w * scale
        scaled_weights[key] = w_new
        sum_abs += abs(w_new)
        
    if sum_abs > 0.0:
        for key in scaled_weights:
            scaled_weights[key] = round(scaled_weights[key] / sum_abs, 4)
        return scaled_weights

    return weights

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
    방향 뿐만 아니라 예측 %가 오전 9시 실제 시초가 등락률(%)에 가까울수록 가중치를 더 많이 증대하며,
    분석 실행 시점이 오전 9시 시초가 결정 시점에 가까울수록 시간 가중치(time_weight)를 부여해 가중치 학습 속도를 증폭합니다.
    """
    actual_open = entry.get("actual_open")
    current_price = entry.get("current_price")
    if actual_open is not None and current_price is not None and current_price > 0:
        actual_change_pct = ((actual_open - current_price) / current_price) * 100
    else:
        actual_change_pct = 0.0 if actual_dir == "FLAT" else 0.5 if actual_dir == "UP" else -0.5

    # 0. 분석 실행 시각과 시초가 결정 시점(오전 9시) 간의 차이에 따른 시간 가중치(time_weight) 계산
    time_weight = 1.0
    try:
        pred_date_str = entry["date"].split(" ")[0]
        pred_dt = datetime.datetime.strptime(pred_date_str, "%Y-%m-%d").date()
        
        try:
            pred_time_str = entry["date"].split(" ")[1]
            pred_time = datetime.datetime.strptime(pred_time_str, "%H:%M:%S").time()
        except Exception:
            pred_time = datetime.time(15, 0, 0)
            
        if pred_time < datetime.time(9, 0, 0):
            target_date = pred_dt
        else:
            d = pred_dt + datetime.timedelta(days=1)
            while d.weekday() >= 5:  # 주말 건너뛰기
                d += datetime.timedelta(days=1)
            target_date = d
            
        pred_time_dt = datetime.datetime.strptime(entry["date"], "%Y-%m-%d %H:%M:%S")
        target_9am = datetime.datetime.combine(target_date, datetime.time(9, 0, 0))
        
        if pred_time_dt > target_9am:
            t_diff_hours = 0.0
        else:
            t_diff_hours = (target_9am - pred_time_dt).total_seconds() / 3600.0
            
        # 1시간 이내에 분석한 경우 (초접근): 1.3배
        if t_diff_hours <= 1.0:
            time_weight = 1.3
        # 3시간 이내: 1.1배
        elif t_diff_hours <= 3.0:
            time_weight = 1.1
        # 12시간 이내: 0.8배
        elif t_diff_hours <= 12.0:
            time_weight = 0.8
        # 12시간 초과: 0.5배
        else:
            time_weight = 0.5
    except Exception as e:
        print(f"[Warning] 시간 가중치 연산 실패: {e}")
        time_weight = 1.0

    # 오차 크기 및 방향 일치도에 기반한 동적 리워드 계산 함수 (내부 헬퍼)
    def calculate_dynamic_reward(pred_val: float, actual_val: float) -> float:
        err = abs(pred_val - actual_val)
        pred_dir = 1.0 if pred_val > 0 else -1.0 if pred_val < 0 else 0.0
        act_dir = 1.0 if actual_val > 0 else -1.0 if actual_val < 0 else 0.0
        
        # 방향이 일치하고 둘 다 FLAT이 아닐 때
        direction_match = (pred_dir == act_dir) and (pred_dir != 0.0)
        
        if direction_match:
            # 오차가 시초가에 가까울수록(작을수록) 가중치 대폭 증대
            if err <= 0.2:
                return 0.05
            elif err <= 0.5:
                return 0.03
            elif err <= 1.0:
                return 0.01
            else:
                return 0.00
        else:
            # 방향이 틀렸거나 오차가 큰 경우 벌점 부여
            if err > 1.5:
                return -0.04
            elif err > 1.0:
                return -0.03
            elif err <= 0.5:
                return -0.01  # 오차 자체가 매우 작다면 방향이 틀려도 최소 페널티
            else:
                return -0.02

    # 1. Consensus 가중치 학습
    consensus_weights = entry.get("before_consensus_weights")
    if not consensus_weights:
        consensus_weights = _get_configured_consensus_weights()
        
    components = entry.get("components")
    if not components:
        return {}
        
    updated_c = consensus_weights.copy()
    
    # 각 영역별 실제 등락률 대비 오차를 반영하여 동적 보상 결정
    rewards_c = {}
    for comp, val in components.items():
        base_reward = calculate_dynamic_reward(val, actual_change_pct)
        rewards_c[comp] = base_reward * time_weight
        
    for comp in updated_c:
        updated_c[comp] += rewards_c.get(comp, 0.0)
        updated_c[comp] = max(0.01, updated_c[comp])
        
    total_c = sum(updated_c.values())
    if total_c > 0:
        for comp in updated_c:
            updated_c[comp] = round(updated_c[comp] / total_c, 4)
            
    # 2. 매크로 가중치 학습
    macro_weights = entry.get("before_macro_weights")
    if not macro_weights:
        macro_weights = _get_configured_macro_weights(entry.get("date"), entry.get("macro_data"))
        
    macro_data = entry.get("macro_data")
    updated_m = macro_weights.copy()
    
    if macro_data:
        for key, val in macro_data.items():
            base_w = BASE_MACRO_WEIGHTS.get(key, 0.0)
            sign = 1.0 if base_w >= 0 else -1.0
            
            # 음수 상관관계 지표(환율, VIX 등)의 경우 변동 부호를 반전시켜 주가 영향력으로 치환
            pred_macro_pct = val * sign
            
            base_reward = calculate_dynamic_reward(pred_macro_pct, actual_change_pct)
            # 매크로 지표 개수가 많으므로 업데이트 단위 비율을 조절(0.05배 적용)
            reward = base_reward * time_weight * 0.05
            
            w_i = updated_m.get(key, 0.0)
            if base_w >= 0:
                updated_m[key] += reward
                updated_m[key] = max(0.0, updated_m[key])
            else:
                updated_m[key] -= reward
                updated_m[key] = min(0.0, updated_m[key])
                
        total_abs = sum(abs(v) for v in updated_m.values())
        if total_abs > 0:
            for key in updated_m:
                updated_m[key] = round(updated_m[key] / total_abs, 4)
                
    valid_keys = set(MACRO_WEIGHTS.keys())
    updated_m = {k: v for k, v in updated_m.items() if k in valid_keys}
    for k, v in MACRO_WEIGHTS.items():
        if k not in updated_m:
            updated_m[k] = v

    # 3. AI 모델 개별 가중치 학습
    ai_weights = entry.get("before_ai_weights")
    if not ai_weights:
        ai_weights = _get_configured_ai_weights()
        
    ai_predictions = entry.get("ai_predictions")
    updated_a = ai_weights.copy()
    
    if ai_predictions:
        rewards_a = {}
        for mdl, pred in ai_predictions.items():
            pred_val = pred.get("change_pct", 0.0)
            base_reward = calculate_dynamic_reward(pred_val, actual_change_pct)
            rewards_a[mdl] = base_reward * time_weight
            
        for mdl in updated_a:
            updated_a[mdl] += rewards_a.get(mdl, 0.0)
            updated_a[mdl] = max(0.01, updated_a[mdl])
            
        total_a = sum(updated_a.values())
        if total_a > 0:
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
   - Kodex200: {m['Kodex200']['value']:,} ({m['Kodex200']['change_pct']}%)
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
   - 나스닥(0%): {m['NASDAQ']['value']:,} ({m['NASDAQ']['change_pct']}%)

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
  "reason": "예측의 근거를 등락 결정 비중(가중치)이 높은 요인 순서대로 머리글을 붙여 각각 짧은 한 줄 문장으로 요약하여 반드시 10개 이상의 불릿 포인트로 작성하십시오. 각 줄은 반드시 검은색 원형 불릿 기호(•)로 시작해야 합니다. (예: '• [코스피 선물] ...\\n• [나스닥 선물] ...\\n• [Kodex200] ...\\n• [원/달러 환율] ...\\n• [삼성전자] ...') 문장이 너무 길어지지 않게 간결하고 직관적으로 작성해 주십시오."
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
            score += m["Kodex200"]["change_pct"] * 0.6
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
        
        # 각 모델별 관심 매크로 지표 정의 (우선순위 후보군)
        model_candidates = {
            "Gemini": ["Kospi_Future", "Samsung", "Hynix", "Technical_Analysis1", "Technical_Analysis2", "SOX_Index", "MSCI_Korea", "USD_KRW", "VIX_Index", "NASDAQ", "US10Y_Treasury", "Short_Selling"],
            "ChatGPT": ["Kospi_Future", "Nasdaq_Future", "Kodex200", "VIX_Index", "Technical_Analysis2", "SOX_Index", "NASDAQ", "USD_KRW", "Bitcoin", "Gold_Future", "US_CPI", "US_Rate"],
            "Claude": ["US_CPI", "USD_KRW", "USD_JPY", "US10Y_Treasury", "Samsung", "KR_Rate", "KR_Bond", "US_Rate", "SOX_Index", "Famous_Remarks", "Nikkei_225", "Shanghai_Composite"],
            "Grok": ["VIX_Index", "WTI_Crude", "Kospi_Future", "USD_KRW", "Bitcoin", "Technical_Analysis1", "Famous_Remarks", "Nasdaq_Future", "NASDAQ", "Gold_Future", "Dollar_Index", "MSCI_Korea"]
        }
        
        candidates = model_candidates.get(model_name, ["Kospi_Future", "Samsung", "Technical_Analysis1"])
        
        # 각 후보 지표의 중요도 계산: 가중치 * 절대 변동률
        # 주가(Samsung, Hynix)는 매크로 가중치가 없으므로 고정 중요도(0.1)를 부여
        import pandas as pd
        scored_candidates = []
        macro_weights = _get_configured_macro_weights(data.get("timestamp"), data.get("macro"))
        
        for factor in candidates:
            w = abs(macro_weights.get(factor, 0.05))
            if factor == "Samsung":
                val = hw["Samsung"]["change_pct"]
                imp = 0.12 * abs(val)
            elif factor == "Hynix":
                val = hw["Hynix"]["change_pct"]
                imp = 0.08 * abs(val)
            else:
                val = m.get(factor, {}).get("change_pct", 0.0)
                imp = w * abs(val)
            scored_candidates.append((imp, factor, val))
            
        # 중요도(imp) 순으로 내림차순 정렬 (중요도 순서대로 나열)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        bullet_points = []
        for imp, factor, val in scored_candidates[:10]:
            # 각 지표에 따른 다이나믹 텍스트 생성
            if factor == "Kospi_Future":
                txt = f"• [코스피 선물] 야간 코스피 200 선물 지수가 {val:+.2f}% 변동하여 개장 초반 지수 출발 강도 결정"
            elif factor == "Nasdaq_Future":
                txt = f"• [나스닥 선물] 미국 나스닥 100 선물 지수({val:+.2f}%)의 추세가 국내 기술주 투자 심리에 영향"
            elif factor == "Kodex200":
                txt = f"• [Kodex200] 전일 KODEX 200 주가 흐름({val:+.2f}%)이 추세 지속 및 개장 동시호가 예측의 기저로 작용"
            elif factor == "Samsung":
                txt = f"• [삼성전자] 시총 1위 삼성전자({val:+.2f}%)의 전일 마감 강도가 지수 수급 지지력을 지탱"
            elif factor == "Hynix":
                txt = f"• [SK하이닉스] SK하이닉스({val:+.2f}%)의 강세/약세가 국내 반도체 섹터 매수 강도 결정"
            elif factor == "USD_KRW":
                txt = f"• [원/달러 환율] 원/달러 환율 변동률({val:+.2f}%)에 따른 외인 현선물 수급 유출입 압력 자극"
            elif factor == "USD_JPY":
                txt = f"• [엔/달러 환율] 엔/달러 환율 변동({val:+.2f}%)이 아시아 제조 대형주 및 수출 경쟁구도에 영향"
            elif factor == "VIX_Index":
                txt = f"• [공포지수 VIX] 글로벌 변동성 VIX 지수가 {val:+.2f}% 변동하여 위험자산 선호 심리 자극"
            elif factor == "SOX_Index":
                txt = f"• [필라델피아 반도체] 필라델피아 반도체 지수의 변동률({val:+.2f}%)이 반도체 대장주 장초반 방향성 견인"
            elif factor == "Technical_Analysis1":
                txt = f"• [기술적 분석1] KODEX 200 최근 5일 이동평균선 대비 괴리율({val:+.2f}%)의 기술적 지지/저항 강도"
            elif factor == "Technical_Analysis2":
                txt = f"• [기술적 분석2] RSI/MACD 종합 보조 분석 지표 변동률({val:+.2f}%)에 의한 기술적 매수 점수 반영"
            elif factor == "WTI_Crude":
                txt = f"• [국제 유가] WTI 원유 선물 가격이 {val:+.2f}% 변동하여 인플레이션 및 제조 기업 원가 우려에 작용"
            elif factor == "US_CPI":
                txt = f"• [미국 CPI] 미국 소비자물가지수(CPI) 변동률({val:+.2f}%)에 따른 글로벌 기준금리 시나리오 반영"
            elif factor == "US10Y_Treasury":
                txt = f"• [미 국채금리] 미 10년물 국채 금리 변동성({val:+.2f}%)이 성장주 멀티플 및 밸류에이션 부담 유발"
            elif factor == "KR_Rate":
                txt = f"• [국내 금리] 한국은행 기준금리 변동률({val:+.2f}%)이 기업 자금 조달 비용 및 시중 유동성 영향"
            elif factor == "KR_Bond":
                txt = f"• [국내 채권금리] 국고채 3년물 금리 변동성({val:+.2f}%)이 자본시장 할인율 부담으로 작동"
            elif factor == "MSCI_Korea":
                txt = f"• [MSCI 한국] MSCI Korea Index의 {val:+.2f}% 변동이 외인 패시브 자금의 장초반 방향성에 선행 작동"
            elif factor == "NASDAQ":
                txt = f"• [나스닥 지수] 미국 나스닥 종합 지수의 전일 마감 변동률({val:+.2f}%)이 국내 성장 섹터 심리 자극"
            elif factor == "Bitcoin":
                txt = f"• [디지털 자산] 비트코인 가격이 {val:+.2f}% 변동하여 전반적인 위험자산 투자 심리(Sentiment)를 대변"
            elif factor == "Famous_Remarks":
                txt = f"• [유명인사 발언] 연준 위원들의 매파/비둘기파 성향 발언 변동 영향({val:+.2f}%)에 의한 변동성 유발"
            else:
                txt = f"• [{factor}] 지표의 실시간 {val:+.2f}% 변동폭이 개장 동시호가 매수 강도 결정에 작용"
            bullet_points.append(txt)
            
        if clean_rumor and clean_rumor != "증권가 풍문 요인 미비":
            bullet_points.append(f"• [증권가 루머] {clean_rumor} 소식에 따른 특정 섹터/테마 수급 변화 심리 작용")
        else:
            bullet_points.append(f"• [증권가 루머] 증권가 사설 정보지(찌라시) 상의 수급 설설(설)에 따른 투자 심리 눈치보기")
            
        reason_text = "\n".join(bullet_points)
        
        return {
            "direction": direction,
            "change_pct": predicted_pct,
            "target_price": target_price,
            "reason": reason_text
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
            
            # 10개 이상 항목 강제 제한 및 보정
            lines = [line.strip() for line in reason.split("\n") if line.strip().startswith("•") or line.strip().startswith("-")]
            if len(lines) < 10:
                mock_pred = self._get_mock_prediction(model_name, data)
                mock_lines = [l.strip() for l in mock_pred["reason"].split("\n") if l.strip()]
                for ml in mock_lines:
                    if len(lines) >= 11:
                        break
                    if ml not in lines:
                        lines.append(ml)
            elif len(lines) > 15:
                lines = lines[:15]
            reason = "\n".join(lines)
            
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
        macro_weights = _get_configured_macro_weights(data.get("timestamp"), data.get("macro"))
        
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
            "Kodex200": {"value": 325400.0, "change_pct": -0.45},
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
            "NASDAQ": {"value": 16000.00, "change_pct": 0.50}
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


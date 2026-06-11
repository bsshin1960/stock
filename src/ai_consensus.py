# -*- coding: utf-8 -*-
import os
import re
import json
import random
import traceback
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
from src.config import DEFAULT_WEIGHTS

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
        weights_str = ", ".join([f"{model}: {weight*100}%" for model, weight in DEFAULT_WEIGHTS.items()])

        prompt = f"""
당신은 국내 최정상급 금융공학 퀀트 애널리스트이자 글로벌 매크로 주식 전문가입니다.
제공된 시장 정보를 바탕으로 **KODEX 200 (코스피 200 지수 ETF)**의 **내일 시초가(Open Price)**가 **금일 오후 3시 현재가({k['current_price']:,}원)** 대비 어떻게 변화할지 분석하십시오.

[분석 기초 데이터]
1. KODEX 200 및 영향력 대형주 현황:
   - KODEX 200 금일 3시 현재가: {k['current_price']:,} 원 (전일 대비 {k['change_pct']}% 변동)
   - 삼성전자: {hw['Samsung']['price']:,} 원 (전일 대비 {hw['Samsung']['change_pct']}% 변동)
   - SK하이닉스: {hw['Hynix']['price']:,} 원 (전일 대비 {hw['Hynix']['change_pct']}% 변동)
   - KODEX 200 기술적 지표: 
     * 5일 이동평균선: {k['sma5']:,} 원 │ 20일선: {k['sma20']:,} 원 │ 60일선: {k['sma60']:,} 원
     * RSI(14): {k['rsi14']} (과매수/과매도)
     * MACD: {k['macd']} (Signal: {k['macd_signal']}, Hist: {k['macd_hist']})
     * 볼린저 밴드: 상한선 {k['bb_upper']:,} 원 │ 하한선 {k['bb_lower']:,} 원

2. 선물 시장 및 글로벌 거시 경제, 일본 상황 (공포 지수 포함):
   - 코스피 200 선물: {m['Kospi_Future']['value']:,} ({m['Kospi_Future']['change_pct']}%)
   - 나스닥 100 선물: {m['Nasdaq_Future']['value']:,} ({m['Nasdaq_Future']['change_pct']}%)
   - S&P 500 선물: {m['SP500_Future']['value']:,} ({m['SP500_Future']['change_pct']}%)
   - 원/달러 환율: {m['USD_KRW']['value']:,} 원 ({m['USD_KRW']['change_pct']}%)
   - 달러/엔 환율 (일본 상황): {m['USD_JPY']['value']:,} 엔 ({m['USD_JPY']['change_pct']}%)
   - 일본 닛케이 225 지수: {m['Nikkei_225']['value']:,} ({m['Nikkei_225']['change_pct']}%)
   - CBOE VIX (공포 지수): {m['VIX_Index']['value']:,} ({m['VIX_Index']['change_pct']}%)
   - 미국 10년물 국채 금리: {m['US10Y_Treasury']['value']}% ({m['US10Y_Treasury']['change_pct']}%)
   - WTI 원유 선물: {m['WTI_Crude']['value']}$ ({m['WTI_Crude']['change_pct']}%)

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
  "reason": "여기에 대형주 동향, 코스피/미국 선물 흐름, 엔화/닛케이 등 일본 상황이 한국 시장에 미치는 영향, 공포지수(VIX), 뉴스 및 증권가 소문을 논리적으로 연동하여 2~3문장의 분석적이고 신뢰도 높은 예측 근거를 작성하십시오."
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
        
        # 신규 변수들을 포함한 정밀한 시뮬레이션 지수 계산
        score = 0.0
        
        # 1. 미국/한국 선물 변동 반영 (매우 중요)
        score += m["Kospi_Future"]["change_pct"] * 1.8
        score += m["Nasdaq_Future"]["change_pct"] * 1.2
        
        # 2. 영향력 대형주 반영
        score += hw["Samsung"]["change_pct"] * 0.8
        score += hw["Hynix"]["change_pct"] * 0.4
        
        # 3. 환율 및 공포 지수 반영 (환율/VIX 상승은 악재)
        score -= m["USD_KRW"]["change_pct"] * 1.0
        score -= (m["VIX_Index"]["value"] - 15.0) * 0.05  # VIX가 15 이상이면 하락 가속
        
        # 4. 일본 상황 반영 (엔/달러 상승(엔화 약세)은 수출 경합도 측면에서 한국 시장에 부정적, 닛케이 상승은 동조화 긍정적)
        score -= m["USD_JPY"]["change_pct"] * 0.4
        score += m["Nikkei_225"]["change_pct"] * 0.5
        
        rsi = k["rsi14"] or 50
        if rsi < 30:
            score += 0.6
        elif rsi > 70:
            score -= 0.6
            
        # 모델별 고유 편차(Seed 고정)
        random.seed(hash(model_name + str(current_price)))
        model_variance = random.uniform(-0.25, 0.25)
        
        # 루머/이슈에 따른 가중 편차 (특정 단어 존재 시 가산/감산)
        rumor_sentiment = 0.0
        for rum in rumors:
            if "공급" in rum or "호조" in rum or "유입" in rum or "비둘기" in rum:
                rumor_sentiment += 0.15
            if "우려" in rum or "매파" in rum or "감소" in rum or "차질" in rum:
                rumor_sentiment -= 0.15
                
        score += rumor_sentiment
        predicted_pct = round((score * 0.6) + model_variance, 2)
        predicted_pct = max(-3.0, min(3.0, predicted_pct))
        
        direction = "UP" if predicted_pct >= 0 else "DOWN"
        target_price = int(current_price * (1 + predicted_pct / 100))
        
        # 풍문/소문을 분석 이유에 임베딩
        selected_rumor = rumors[0] if rumors else "증권가 풍문 요인 미비"
        clean_rumor = selected_rumor.replace("[단독 루머] ", "").replace("[소문] ", "").replace("[찌라시] ", "").replace("[반도체설] ", "").replace("[수급 소문] ", "")
        
        reasons = {
            "Gemini": f"코스피/나스닥 선물의 동조화 및 일본 닛케이 225({m['Nikkei_225']['change_pct']}%) 추세를 모델링했습니다. 엔/달러 환율이 {m['USD_JPY']['value']}엔 수준으로 변동성을 보이며 아시아 수급 경합도가 강해진 가운데, 삼성전자의 {hw['Samsung']['change_pct']}% 반등세와 VIX 지수를 고려하여 내일 시가는 {predicted_pct}% 변동한 {target_price:,}원으로 귀결될 것입니다.",
            "ChatGPT": f"삼성전자의 흐름과 한미 선물지수, 그리고 일본 엔화 상황(변동률: {m['USD_JPY']['change_pct']}%)을 크로스 분석했습니다. 엔화 가치 약세 여파가 국내 완성차 및 반도체 수출주에 실시간 심리를 제어하고 있으나, 닛케이 지수의 {m['Nikkei_225']['change_pct']}% 흐름과 결합되어 시가는 {predicted_pct}% 변동이 예상됩니다.",
            "Claude": f"거시 경제 지표와 한일 증시 동조성(닛케이: {m['Nikkei_225']['change_pct']}%), 엔/달러 환율({m['USD_JPY']['value']}엔)을 퀀트 분석했습니다. 국내 환율의 {m['USD_KRW']['change_pct']}% 급락세와 소문인 '{clean_rumor}' 요인이 지수 갭개시에 긍정적/부정적 외압으로 혼재되어 시초가는 {target_price:,}원 부근으로 예상됩니다.",
            "Grok": f"VIX({m['VIX_Index']['value']}) 지수에 따른 하방 제한 속에서 엔/달러 환율 변동과 닛케이 225 흐름을 추적했습니다. 대외 경쟁 구도가 심화되는 중이지만 코스피 선물의 견조함으로 {predicted_pct}% 수준의 시가 { '상승' if direction == 'UP' else '하락' } 개장이 주도적입니다."
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
            weights = DEFAULT_WEIGHTS
            
        current_price = data["kodex200"]["current_price"]
        
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
        
        # 각 매크로 지표의 실질 변동률
        factors = {
            "Kospi_Future": m["Kospi_Future"]["change_pct"],
            "Nasdaq_Future": m["Nasdaq_Future"]["change_pct"],
            "SP500_Future": m["SP500_Future"]["change_pct"],
            "USD_KRW": m["USD_KRW"]["change_pct"],
            "USD_JPY": m["USD_JPY"]["change_pct"],
            "Nikkei_225": m["Nikkei_225"]["change_pct"],
            "VIX_Index": m["VIX_Index"]["change_pct"],
            "US10Y_Treasury": m["US10Y_Treasury"]["change_pct"],
            "WTI_Crude": m["WTI_Crude"]["change_pct"],
        }
        
        macro_weighted_sum = 0.0
        abs_macro_weight_sum = 0.0
        
        for key, val in factors.items():
            w = MACRO_WEIGHTS.get(key, 0.0)
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

        # 5단계: 최종 종합 가중 합산 (AI 50% + 매크로 30% + 뉴스 15% + 소문 5%)
        w_ai = CONSENSUS_WEIGHTS.get("AI_Consensus", 0.50)
        w_macro = CONSENSUS_WEIGHTS.get("Macro_Dashboard", 0.30)
        w_news = CONSENSUS_WEIGHTS.get("News_Consensus", 0.15)
        w_rumor = CONSENSUS_WEIGHTS.get("Rumor_Consensus", 0.05)

        consensus_change_pct = (ai_consensus_change_pct * w_ai) + (macro_change_pct * w_macro) + (news_change_pct * w_news) + (rumor_change_pct * w_rumor)
        consensus_change_pct = round(consensus_change_pct, 2)
            
        # 등락 한도 제한 (-3.0% ~ +3.0%)
        consensus_change_pct = max(-3.0, min(3.0, consensus_change_pct))
            
        # 예상 가격 계산 (오늘 3시 가격 기준)
        consensus_target_price = int(current_price * (1 + consensus_change_pct / 100))
        consensus_direction = "UP" if consensus_change_pct >= 0 else "DOWN"
        
        # 합의 종합 분석 리포트 생성
        reasons_summary = "   │   ".join([f"{model}: {res['change_pct']}%" for model, res in ai_results.items()])
        
        consensus_weights_desc = f"AI {int(w_ai*100)}% │ 매크로 {int(w_macro*100)}% │ 뉴스 {int(w_news*100)}% │ 소문 {int(w_rumor*100)}%"
        
        consensus_reason = (
            f"4대 AI 분석 의견({int(w_ai*100)}%), 글로벌 매크로 지표({int(w_macro*100)}%), 실시간 속보 뉴스({int(w_news*100)}%), 증권가 소문/이슈({int(w_rumor*100)}%)를 종합 가중 분석한 결론입니다.\n"
            f"AI 예측 변동률({ai_consensus_change_pct:+.2f}%), 글로벌 매크로 평균 변동률({macro_change_pct:+.2f}%), "
            f"실시간 뉴스 분석({news_change_pct:+.2f}%), 소문 감성 지표({rumor_change_pct:+.2f}%)가 반영되었으며,\n"
            f"내일 KODEX 200 시초가는 금일 대비 {consensus_change_pct:+.2f}% 변동한 {consensus_target_price:,}원 부근 형성이 유력합니다."
        )
        
        return {
            "direction": consensus_direction,
            "change_pct": consensus_change_pct,
            "target_price": consensus_target_price,
            "reason": consensus_reason,
            "details": reasons_summary
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
            "Nikkei_225": {"value": 38720.50, "change_pct": -0.42},
            "US10Y_Treasury": {"value": 4.432, "change_pct": -0.85},
            "WTI_Crude": {"value": 78.45, "change_pct": 0.52},
            "VIX_Index": {"value": 13.45, "change_pct": 1.25}
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


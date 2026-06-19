# -*- coding: utf-8 -*-
import os
import datetime
from pathlib import Path
from src.config import REPORTS_DIR
from src.ai_consensus import _get_configured_ai_weights

class PredictionReporter:
    """예측 결과를 마크다운 파일로 저장하고 이력을 관리하는 클래스"""

    def __init__(self):
        pass

    def generate_report(self, data: dict, ai_results: dict, consensus: dict) -> Path:
        """분석 결과를 깔끔한 마크다운 보고서로 작성하여 파일로 저장"""
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        file_date_str = now.strftime("%Y%m%d_%H%M%S")
        
        k = data["kodex200"]
        hw = data["heavyweights"]
        m = data["macro"]
        
        # 1. 상승/하락 심볼 결정
        dir_symbol = "▲ 상승" if consensus["direction"] == "UP" else "▼ 하락"
        
        # 가중치 세부 정보 (안전한 Fallback 처리)
        weights = consensus.get("weights", _get_configured_ai_weights())
        weights_detail = " | ".join([f"{model}({weight*100:.1f}%)" for model, weight in weights.items()])
        
        # 개별 AI 의견 요약 테이블 빌드 (동적 가중치 반영)
        ai_rows = ""
        for mdl, weight in weights.items():
            res = ai_results.get(mdl, {"change_pct": 0.0, "target_price": 0, "reason": "의견 없음"})
            ai_rows += f"| **{mdl}** | `{weight*100:.1f}%` | `{res['change_pct']}%` | `{res['target_price']:,}원` | {res['reason']} |\n"
        
        # 분석 실행 시간(오전 9시 이전 vs 이후)에 따라 예측 대상 시점의 명칭을 동적으로 결정
        try:
            time_part = timestamp_str.split(" ")[1]
            hour = int(time_part.split(":")[0])
        except Exception:
            hour = datetime.datetime.now().hour
            
        if hour < 9:
            target_title = "오전 9시"
            target_desc = "오늘(당일) 오전 9시"
            current_desc = "현재가"
            base_time_desc = "금일 현재가"
            analysis_type = "장 시작 전 분석용"
        else:
            target_title = "내일"
            target_desc = "내일"
            current_desc = "금일 오후 3시 현재가"
            base_time_desc = "오늘 3시 현재가"
            analysis_type = "장 마감 분석용"

        report_content = f"""# KODEX 200 {target_title} 시가 예측 보고서
**분석 일시**: `{timestamp_str}` ({analysis_type})

---

## 1. 종합 예측 결론 (Consensus)
> **[최종 결론]**
> {target_desc} KODEX 200 시초가는 **{current_desc} 대비 {dir_symbol}** 할 것으로 전망됩니다.
> 
> * **예측 등락률**: **`{consensus['change_pct']}%`**
> * **예상 시초가**: **`{consensus['target_price']:,} 원`**
> * **{base_time_desc}**: `{k['current_price']:,} 원` (전일비 {k['change_pct']}%)
> * **예상 변동폭**: `{int(consensus['target_price'] - k['current_price']):+,} 원`
> * **종합 가중치 설정**: `{weights_detail}`

### 종합 분석 의견
{consensus['reason']}

---

## 2. 개별 AI 분석 의견 요약 (반영 비율 가중치 합산)
| AI 모델 | 반영 가중치 | 예측 등락률 | 예상 시초가 | 핵심 요약 및 논거 |
| :--- | :---: | :---: | :---: | :--- |
{ai_rows}
---

## 3. 분석 시점 시장 데이터 현황

        ### A. 주가 변동 요인
        - **KODEX 200 현재가**: `{k['current_price']:,} 원` (전일비 {k['change_pct']}%)
        - **삼성전자**: `{hw['Samsung']['price']:,} 원` (전일비 {hw['Samsung']['change_pct']}%)
        - **SK하이닉스**: `{hw['Hynix']['price']:,} 원` (전일비 {hw['Hynix']['change_pct']}%)
        - **KODEX 200 기술적 지표**:
          - 이동평균선: 5일선 `{k['sma5']:,}원` | 20일선 `{k['sma20']:,}원` | 60일선 `{k['sma60']:,}원`
          - RSI (14): `{k['rsi14']}` | MACD Hist: `{k['macd_hist']}`
          - 볼린저 밴드: 상한 `{k['bb_upper']:,}원` | 하한 `{k['bb_lower']:,}원`
        
        ### B. 주가 변경 인자 (30%)
        | 지표명 | 실시간 값 | 변동률(%) | 한국 주가 영향도 |
        | :--- | :---: | :---: | :---: |
        | **코스피 200 선물** | `{m['Kospi_Future']['value']:,}` | `{m['Kospi_Future']['change_pct']}%` | 직접 영향 |
        | **나스닥 100 선물** | `{m['Nasdaq_Future']['value']:,}` | `{m['Nasdaq_Future']['change_pct']}%` | 방향 결정 |
        | **Kodex200** | `{m['Kodex200']['value']:,}원` | `{m['Kodex200']['change_pct']}%` | 추세 동조 |
        | **원/달러 환율** | `{m['USD_KRW']['value']:,}원` | `{m['USD_KRW']['change_pct']}%` | 수급 영향 |
        | **엔/달러 환율 (일본)** | `{m['USD_JPY']['value']:,}엔` | `{m['USD_JPY']['change_pct']}%` | 경쟁력 영향 |
        | **미국소비자물가지수** | `{m['US_CPI']['value']:,}` | `{m['US_CPI']['change_pct']}%` | 인플레이션 요인 |
        | **VIX 공포 지수** | `{m['VIX_Index']['value']:,}` | `{m['VIX_Index']['change_pct']}%` | 시장 위험 지표 |
        | **미국 10년물 국채금리** | `{m['US10Y_Treasury']['value']}%` | `{m['US10Y_Treasury']['change_pct']}%` | 밸류에이션 부담 |
        | **WTI 국제 유가** | `{m['WTI_Crude']['value']}$` | `{m['WTI_Crude']['change_pct']}%` | 비용 부담 요인 |
        | **한국은행 기준금리 (국내 금리)** | `{m['KR_Rate']['value']}%` | `{m['KR_Rate']['change_pct']}%` | 조달 비용 요인 |
        | **국고채 3년 금리 (국내 채권금리)** | `{m['KR_Bond']['value']}%` | `{m['KR_Bond']['change_pct']}%` | 시중 금리 요인 |
        | **필라델피아 반도체** | `{m['SOX_Index']['value']:,}` | `{m['SOX_Index']['change_pct']}%` | 반도체 업황 |
        | **달러 인덱스** | `{m['Dollar_Index']['value']:,}` | `{m['Dollar_Index']['change_pct']}%` | 글로벌 달러 강세 |
        | **금 선물** | `{m['Gold_Future']['value']:,}$` | `{m['Gold_Future']['change_pct']}%` | 안전자산 선호 |
        | **기술적분석1 (Kodex200 주간주가분석)** | `-` | `-` | 분석 보조 요인 |
        | **기술적분석2 (Kodex200 MA,MACD,RSI)** | `-` | `-` | 분석 보조 요인 |
        | **일본 닛케이 225** | `{m['Nikkei_225']['value']:,}` | `{m['Nikkei_225']['change_pct']}%` | 아시아 증시 영향 |
        | **상해 종합지수** | `{m['Shanghai_Composite']['value']:,}` | `{m['Shanghai_Composite']['change_pct']}%` | 중국 경기 영향 |
        | **MSCI 한국 ETF 종가** | `{m['MSCI_Korea']['value']:,}` | `{m['MSCI_Korea']['change_pct']}%` | 한국 증시 선행 |
        | **상위종목공매도** | `{m['Short_Selling']['value']:,}` | `{m['Short_Selling']['change_pct']}%` | 시장 수급 요인 |
        | **유명인사 발언** | `{m['Famous_Remarks']['value']:,}` | `{m['Famous_Remarks']['change_pct']}%` | 투자 심리 영향 |
        | **비트코인** | `{m['Bitcoin']['value']:,}$` | `{m['Bitcoin']['change_pct']}%` | 디지털 자산 센티먼트 |
        | **미국 기준금리 (미국 금리) (0%)** | `{m['US_Rate']['value']:.2f}%` | `{m['US_Rate']['change_pct']}%` | 향후 추가 예정 |
        | **나스닥(0%)** | `{m['NASDAQ']['value']:,}` | `{m['NASDAQ']['change_pct']}%` | 향후 추가 예정 |
        
        ### C. 당일 시장 주요 뉴스 헤드라인
        """
        for title in data["news"]:
            report_content += f"- {title}\n"
            
        report_content += "\n### D. 증권가 소문/이슈\n"
        for rumor in data["rumors"]:
            report_content += f"- *{rumor}*\n"
            
        report_content += """
---
*본 보고서는 4대 AI의 의견을 종합하여 생성된 보조적 주가 예측 자료이며, 모든 투자의 결정과 책임은 투자자 본인에게 있습니다.*
"""
        # 불러오기 기능을 위한 JSON RAW_DATA 임베딩
        import json
        raw_state = {
            "current_data": data,
            "ai_results": ai_results,
            "consensus_result": consensus
        }
        json_str = json.dumps(raw_state, ensure_ascii=False)
        report_content += f"\n<!-- RAW_DATA: {json_str} -->\n"

        # 파일 저장
        filename = f"KODEX200_Prediction_{file_date_str}.md"
        filepath = REPORTS_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        print(f"[System] 보고서가 저장되었습니다: {filepath}")
        return filepath

if __name__ == "__main__":
    pass

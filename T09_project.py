import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import html
import re


# 1. Hugging Face FinBERT 감성 분석 모델 로드 (캐싱으로 속도 최적화)
@st.cache_resource
def load_sentiment_model():
    return pipeline("text-classification", model="snunlp/KR-FinBert-SC")


try:
    classifier = load_sentiment_model()
except Exception as e:
    st.error(f"모델 로드 중 오류가 발생했습니다: {e}")

# --- 사이드바 네이버 API 키 설정 및 인증 확인 ---
st.sidebar.header("🔑 네이버 API 설정")
st.sidebar.markdown("""
네이버 개발자 센터에서 발급받은 본인의 API 키를 입력해주세요.  
[네이버 개발자 센터 바로가기](https://developers.naver.com/)
""")
user_client_id = st.sidebar.text_input("Naver Client ID", type="password", placeholder="Client ID를 입력하세요")
user_client_secret = st.sidebar.text_input("Naver Client Secret", type="password", placeholder="Client Secret을 입력하세요")

# 인증 키 상태 안내 및 확인 문구
if user_client_id and user_client_secret:
    st.sidebar.success("🔑 인증키 입력 완료")
else:
    st.sidebar.info("📢 뉴스 기사 제목 감성 분석을 위해 네이버 API 인증 키를 입력해 주세요.")

# 스트림릿 페이지 기본 설정
st.set_page_config(layout="wide", page_title="주식 감성 분석 및 추천")
st.title("📈 초보 투자자를 위한 주식 데이터 & 뉴스 감성 분석")

# 한글 폰트 설정 (Matplotlib 차트 내 한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 좌우 1단 구성 레이아웃 설정
col1, col2 = st.columns([1, 1])

# 세션 상태 초기화 (데이터 보존용)
if 'companies' not in st.session_state:
    st.session_state.companies = []
if 'show_stock' not in st.session_state:
    st.session_state.show_stock = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'current_category' not in st.session_state:
    st.session_state.current_category = ""
if 'stock_df' not in st.session_state:
    st.session_state.stock_df = None
if 'news_plots' not in st.session_state:
    st.session_state.news_plots = {}  # 회사별 매핑 딕셔너리

# --- 왼쪽: 주식 데이터 수집 및 추천 위젯 ---
with col1:
    st.header("📊 주식 데이터 수집")

    category = st.radio("수집 항목 선택", ["거래 상위", "상승", "하락", "시가총액 상위"])

    if st.button("주식 데이터 수집 시작/새로고침"):
        st.session_state.show_stock = True
        st.session_state.current_category = category
        st.session_state.analysis_results = {}
        st.session_state.news_plots = {}
        # 분석이 새로 시작되면 멀티셀렉트 선택 항목도 초기화되도록 처리
        if "selected_companies_widget" in st.session_state:
            st.session_state["selected_companies_widget"] = []

        # 네이버 증권 TOP 종목 URL 맵핑
        url_map = {
            "거래 상위": "https://finance.naver.com/sise/sise_quant.naver",
            "상승": "https://finance.naver.com/sise/sise_rise.naver",
            "하락": "https://finance.naver.com/sise/sise_fall.naver",
            "시가총액 상위": "https://finance.naver.com/sise/sise_market_sum.naver"
        }

        # 크롤링 수행
        res = requests.get(url_map[category], headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        soup = BeautifulSoup(res.text, 'html.parser')

        # 데이터 추출
        stocks = soup.select("table.type_2 tr")
        data = []
        companies = []

        for stock in stocks:
            name_tag = stock.select_one("a.tltle")
            if name_tag and len(data) < 10:  # 상위 10개 수집
                name = name_tag.text.strip()
                tds = stock.select("td")

                if len(tds) < 5:
                    continue

                price = tds[2].text.strip()
                raw_change = tds[3].text.strip().replace("\n", "").replace("\t", "")

                only_number = re.sub(r'[^0-9,.]', '', raw_change)

                if "상승" in raw_change or "상한" in raw_change:
                    change = "+" + only_number
                elif "하락" in raw_change or "하한" in raw_change:
                    change = "-" + only_number
                elif "보합" in raw_change or only_number == "0" or not only_number:
                    change = "0"
                else:
                    change = only_number

                ratio = tds[4].text.strip()

                companies.append(name)
                data.append([name, price, change, ratio])

        st.session_state.companies = companies
        st.session_state.stock_df = pd.DataFrame(data, columns=["회사명", "주가", "금액 변동", "오른 퍼센트"])

    # 주식 데이터 테이블 출력
    if st.session_state.show_stock and st.session_state.stock_df is not None:
        def style_rows(row):
            current_cat = st.session_state.current_category
            if '-' in str(row['금액 변동']) or current_cat == "하락":
                return ['color: blue'] * len(row)
            elif '+' in str(row['금액 변동']) or current_cat == "상승":
                return ['color: red'] * len(row)
            else:
                return ['color: black'] * len(row)


        st.dataframe(st.session_state.stock_df.style.apply(style_rows, axis=1), use_container_width=True)

    # 주식 종목 추천 위젯
    st.write("")
    st.divider()
    st.header("🎯 주식 종목 추천")
    st.caption("실시간 데이터 및 뉴스 감성 분석이 완료된 10개 종목 기반의 추천 결과입니다.")

    btn_recommend = st.button("종목 추천 받기")

    if btn_recommend:
        if not st.session_state.analysis_results:
            st.warning("먼저 우측에서 **'뉴스 제목 감성 분석 시작'**을 완료한 후 버튼을 눌러주세요.")
        else:
            recommended_company = max(
                st.session_state.analysis_results,
                key=lambda k: st.session_state.analysis_results[k]['total_score']
            )

            rec_data = st.session_state.analysis_results[recommended_company]
            cat_name = st.session_state.get('current_category', '선택 항목')

            st.success(
                f"""### 🏆 오늘의 추천 종목: **{recommended_company}**

**[종합 분석 결과 및 추천 근거]**
- 현재 네이버 증권 **'{cat_name}'** 순위 리스트에서 **{rec_data['rank']}위**에 위치하여 투자자들의 높은 관심을 받고 있습니다.
- 실시간 뉴스 제목 분석 결과, **긍정 비율 지표가 {rec_data['positive_ratio']:.1f}%**로 분석 대상 종목 중 시장 심리가 가장 우세합니다. (부정 비율: {rec_data['negative_ratio']:.1f}%)

*※ 본 서비스는 랭킹 순위 점수와 뉴스 감성 스코어(긍정-부정 비율)를 결합한 알고리즘 추천 지표이며, 투자 판단의 참고용으로만 활용하시기 바랍니다.*"""
            )

# --- 오른쪽: 뉴스 제목 감성 분석 ---
with col2:
    st.header("📰 뉴스 제목 감성 분석")

    # 시각화 방식 라디오 버튼
    chart_type = st.radio("시각화 방식 선택", ["가로 막대그래프", "원형(Pie) 그래프"], horizontal=True)

    # 버튼 나란히 배치
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        start_analysis = st.button("뉴스 제목 감성 분석 시작", use_container_width=True)
    with btn_col2:
        clear_analysis = st.button("선택 회사 초기화", use_container_width=True)

    # 초기화 클릭 시: 드롭다운 틀은 남기고 네모 칸(선택 리스트)을 빈 리스트로 초기화
    if clear_analysis:
        st.session_state["selected_companies_widget"] = []
        st.rerun()

    if start_analysis:
        if not user_client_id or not user_client_secret:
            st.error("🔑 왼쪽 사이드바에 **네이버 Client ID**와 **Client Secret**을 모두 입력해주세요.")
        elif not st.session_state.companies:
            st.error("먼저 왼쪽에서 주식 데이터를 수집해 주세요.")
        else:
            with st.spinner("네이버 뉴스 수집 및 딥러닝 감성 분석 중..."):
                headers = {
                    "X-Naver-Client-Id": user_client_id,
                    "X-Naver-Client-Secret": user_client_secret
                }

                temp_results = {}
                temp_plots = {}

                for index, company in enumerate(st.session_state.companies):
                    api_url = f"https://openapi.naver.com/v1/search/news.json?query={company}&display=50"
                    response = requests.get(api_url, headers=headers)

                    if response.status_code != 200:
                        if response.status_code in [401, 403]:
                            st.error(f"**{company}** 호출 실패: API 키가 잘못되었거나 권한이 없습니다. (코드: {response.status_code})")
                        else:
                            st.error(f"**{company}** API 호출 실패 (오류 코드: {response.status_code})")
                        break

                    news_res = response.json()
                    items = news_res.get('items', [])

                    titles = []
                    for item in items:
                        raw_title = item['title']
                        clean_title = re.sub(r'<[^>]*>', '', raw_title)
                        clean_title = html.unescape(clean_title)
                        titles.append(clean_title)

                    if not titles:
                        temp_plots[company] = {"type": "text", "body": f"**{company}**: 관련 최근 기사가 없습니다."}
                        continue

                    # FinBERT 감성 분석 수행
                    results = classifier(titles)

                    counts = {"positive": 0, "neutral": 0, "negative": 0}
                    for r in results:
                        label = r['label'].lower()
                        if label in counts:
                            counts[label] += 1

                    total = len(titles)
                    p_per = (counts["positive"] / total) * 100
                    neu_per = (counts["neutral"] / total) * 100
                    neg_per = (counts["negative"] / total) * 100

                    summary_text = f"**{company} 감정 요약:** 😭 부정 {neg_per:.1f}% | 😐 중립 {neu_per:.1f}% | 🙂 긍정 {p_per:.1f}%"

                    temp_plots[company] = {
                        "type": "plot",
                        "percentages": [neg_per, neu_per, p_per],
                        "text": summary_text
                    }

                    # 알고리즘 점수 계산
                    rank_score = 10 - index
                    sentiment_score = p_per - neg_per
                    total_score = rank_score + sentiment_score

                    temp_results[company] = {
                        "positive_ratio": p_per,
                        "negative_ratio": neg_per,
                        "rank": index + 1,
                        "total_score": total_score
                    }

                st.session_state.news_plots = temp_plots
                st.session_state.analysis_results = temp_results

                # [수정] 분석이 처음 끝나도 강제로 모든 회사를 채워넣지 않고 빈 공간으로 시작 (사용자 선택 유도)
                st.session_state["selected_companies_widget"] = []

    # --- 분석 결과 출력 단 ---
    if st.session_state.news_plots:
        st.write("")
        st.markdown("### 🔍 조회할 회사 선택")
        st.caption("최대 10개까지 복수 선택이 가능합니다. 회사를 선택하면 그래프가 표시됩니다.")

        available_companies = list(st.session_state.news_plots.keys())

        # 위젯 생성 및 세션 바인딩
        selected_companies = st.multiselect(
            "시각화할 회사를 선택하세요",
            options=available_companies,
            key="selected_companies_widget"
        )

        st.divider()

        # 사용자가 선택한 회사가 있을 때만 차트 출력
        for company in selected_companies:
            if company in st.session_state.news_plots:
                item = st.session_state.news_plots[company]

                if item["type"] == "plot":
                    neg_per, neu_per, p_per = item["percentages"]
                    categories = ['부정', '중립', '긍정']
                    colors = ['#ff4d4d', '#dddddd', '#4da6ff']

                    if chart_type == "가로 막대그래프":
                        fig, ax = plt.subplots(figsize=(6, 1.5))
                        bars = ax.barh(categories, [neg_per, neu_per, p_per], color=colors)
                        ax.set_xlim(0, 115)
                        ax.set_title(f"{company} 뉴스 감성 분포 (%)", fontsize=11, fontweight='bold')
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%',
                                    va='center', ha='left', fontsize=9, fontweight='bold')

                    elif chart_type == "원형(Pie) 그래프":
                        fig, ax = plt.subplots(figsize=(4, 4))


                        def make_autopct(values):
                            def my_autopct(pct):
                                return f'{pct:.1f}%' if pct > 0 else ''

                            return my_autopct


                        ax.pie([neg_per, neu_per, p_per], labels=categories, colors=colors,
                               autopct=make_autopct([neg_per, neu_per, p_per]), startangle=90,
                               textprops={'fontsize': 10, 'fontweight': 'bold'})
                        ax.axis('equal')
                        ax.set_title(f"{company} 뉴스 감성 분포", fontsize=12, fontweight='bold')

                    st.pyplot(fig)
                    st.markdown(item["text"])
                    plt.close(fig)
                    st.divider()

                elif item["type"] == "text":
                    st.write(item["body"])
                    st.divider()
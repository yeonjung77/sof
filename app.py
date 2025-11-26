import os
import streamlit as st
from dotenv import load_dotenv
from collections import defaultdict

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from search_timeline import (
    search_keyword_timeline,
    summarize_yearly_insights,
    generate_timeline_synthesis,
)

# ========================================
# 기본 설정
# ========================================
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    st.error("❌ GROQ_API_KEY가 없습니다. .env 또는 Streamlit Secrets에 등록해주세요.")
    st.stop()


# ========================================
# 벡터스토어 로딩
# ========================================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )


# ========================================
# LLM 로딩
# ========================================
@st.cache_resource
def load_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        groq_api_key=groq_key,
    )


vectorstore = load_vectorstore()
llm = load_llm()
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

CHAPTER_LABELS = ["Global Economy", "Consumer Shifts", "Fashion System"]


# ========================================
# 문서 그룹 로딩
# ========================================
@st.cache_resource
def load_grouped_docs():
    all_docs = list(vectorstore.docstore._dict.values())
    by_year_chapter = defaultdict(list)
    by_chapter = defaultdict(list)

    for d in all_docs:
        year = d.metadata.get("year")
        chapter = d.metadata.get("chapter")
        by_year_chapter[(year, chapter)].append(d)
        by_chapter[chapter].append(d)

    return by_year_chapter, by_chapter


by_year_chapter, by_chapter = load_grouped_docs()


# ========================================
# 헬퍼: 문서 포맷팅
# ========================================
def format_docs(docs):
    processed = []
    for d in docs:
        src = os.path.basename(d.metadata.get("source", ""))
        page = d.metadata.get("page", "?")
        year = d.metadata.get("year", "")
        chapter = d.metadata.get("chapter", "")
        header = f"[{year} / {chapter} / {src} p.{page}]"
        processed.append(header + "\n" + d.page_content)
    return "\n\n".join(processed)


# ========================================
# 공통 RAG 프롬프트
# ========================================
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional Fashion MD Research Assistant.\n"
            "Use ONLY the content from McKinsey & BoF 'State of Fashion' (2021–2025).\n"
            "답변은 한국어로, 핵심 용어는 영어 병기해줘.",
        ),
        (
            "human",
            "질문: {question}\n\n"
            "참고 문서:\n{context}",
        ),
    ]
)

qa_chain = qa_prompt | llm | StrOutputParser()


# ========================================
# Streamlit UI 시작
# ========================================
st.set_page_config(page_title="State of Fashion — AI Insight Engine")

st.title("The State of Fashion")
st.title("- AI Insight Engine")
st.caption("AI-powered Insight from SoF 2021–2025 Reports")

st.markdown("---")

# ========================================
# 메인 탭 구성
# ========================================
tab_main, tab_keyword, tab_chapter,tab_country = st.tabs([
    "1️⃣ AI Report Search",
    "2️⃣ Keyword Analytics",
    "3️⃣ Chapter Insighs",
    "4️⃣ Regional Insights",
])


# ============================================================================
# 📌 TAB 1 — 전체 검색 & 질문하기
# ============================================================================
with tab_main:
    st.subheader("Ask Anything — AI Analyzes the Report to Answer Your Questions")

    question = st.text_area("질문 입력", key="qa_question")
    chapter_filter = st.selectbox(
        "검색할 챕터 (옵션)", ["전체"] + CHAPTER_LABELS, index=0
    )

    if st.button("AI에게 질문하기", key="qa_button"):
        if not question.strip():
            st.warning("질문을 입력해주세요.")
        else:
            with st.spinner("보고서를 분석하고 있습니다..."):
                docs = vectorstore.similarity_search(question, k=25)

                if chapter_filter != "전체":
                    docs = [
                        d for d in docs if d.metadata.get("chapter") == chapter_filter
                    ]
                    docs = docs[:8] or docs

                context = format_docs(docs[:8])
                answer = qa_chain.invoke({"question": question, "context": context})

            st.markdown("### 📌 답변")
            st.write(answer)


# ============================================================================
# 📌 TAB 2 — Chapter Insight (서브탭 4개)
# ============================================================================
with tab_chapter:

    sub1, sub2, sub3 = st.tabs(
        [
            "Annual Keyword Insights",
            "Chapter Keyword Timeline",
            "Keyword Mapping"
        ]
    )

    # ---------------------------------------------------
    # 📌 서브탭 1 — 연도별 핵심 키워드
    # ---------------------------------------------------
    with sub1:
        st.subheader("Key Keywords by Year")

        col1, col2 = st.columns(2)
        with col1:
            year = st.selectbox("연도 선택", [2021, 2022, 2023, 2024, 2025])
        with col2:
            chapter = st.selectbox("챕터 선택", CHAPTER_LABELS)

        if st.button("키워드 생성", key="year_chapter_summary_keywords"):
            key = (year, chapter)
            docs = by_year_chapter.get(key, [])

            if not docs:
                st.warning("해당 연도/챕터에 대한 문서를 찾을 수 없습니다.")
            else:
                text = "\n\n".join(d.page_content for d in docs[:20])

                summary_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a senior fashion strategy analyst. "
                            "아래 텍스트를 기반으로 해당 연도/챕터의 핵심 트렌드 키워드를 5개 뽑아 "
                            "각 키워드당 1~2문장 설명을 만들어줘.\n"
                            "설명은 한국어로, 중요한 용어는 영어 병기해줘."
                        ),
                        (
                            "human",
                            "연도: {year}\n챕터: {chapter}\n\n"
                            "분석 텍스트:\n{text}\n\n"
                            "➡ 출력 형식:\n"
                            "Key Insights\n"
                            "- 키워드 1: 설명(1~2줄)\n"
                            "- 키워드 2: 설명\n"
                            "- 키워드 3: 설명\n"
                            "- 키워드 4: 설명\n"
                            "- 키워드 5: 설명"
                        ),
                    ]
                )

                chain = summary_prompt | llm | StrOutputParser()

                with st.spinner("핵심 키워드를 추출하는 중..."):
                    summary = chain.invoke(
                        {"year": year, "chapter": chapter, "text": text}
                    )

                st.write(summary)


    # ---------------------------------------------------
    # 📌 서브탭 2 — 챕터별 키워드 타임라인
    # ---------------------------------------------------
    with sub2:
        st.subheader("Chapter-Based Keyword Timeline Analysis")

        keyword = st.text_input(
            "분석할 키워드 (예: AI, resale, sustainability, Gen Z, silver spenders...)", key="timeline_keyword"
        )
        chapter_sel = st.selectbox(
            "챕터 선택", ["전체"] + CHAPTER_LABELS, index=0, key="timeline_chapter"
        )

        if st.button("타임라인 생성", key="timeline_button"):
            if not keyword.strip():
                st.warning("키워드를 입력해주세요.")
            else:
                ch = None if chapter_sel == "전체" else chapter_sel

                with st.spinner("타임라인 분석 중..."):
                    grouped = search_keyword_timeline(keyword, retriever, chapter=ch)

                    timeline_full = {yr: grouped.get(yr, []) for yr in [2021, 2022, 2023, 2024, 2025]}

                    yearly_summary = {}
                    for yr, docs in timeline_full.items():

                        if not docs:
                            yearly_summary[yr] = "⚠️ 해당 연도에서는 키워드 언급이 거의 없었습니다."
                        else:
                            text = "\n\n".join(docs[:3])
                            prompt = ChatPromptTemplate.from_messages(
                                [
                                    (
                                        "system",
                                        "You are a fashion trend analyst. "
                                        "아래 텍스트에 기반하여 해당 연도의 관점을 2~3문장으로 요약해줘.\n"
                                        "❗ 절대 금지:\n"
                                        "- '2023년의 키워드는 ~입니다' 같은 문장 생성\n"
                                        "- 텍스트에 없는 대표 키워드 생성\n"
                                        "- 패션 트렌드 키워드 선언\n"
                                        "- 해석 지어내기\n"
                                        "❗ 반드시 지킬 것:\n"
                                        "- 텍스트 기반 요약만 생성\n"
                                        "- 한국어로 설명하되 핵심 용어만 영어 병기"
                                    ),
                                    (
                                        "human",
                                        "키워드: {keyword}\n연도: {year}\n\n텍스트:\n{text}"
                                    ),
                                ]
                            )
                            chain = prompt | llm | StrOutputParser()
                            summary = chain.invoke({"keyword": keyword, "year": yr, "text": text})
                            yearly_summary[yr] = summary

                    synthesis_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You are a senior fashion strategist."
                                "연도별 분석 내용을 기반으로 전체 흐름을 딱 3문장으로 요약.\n"
                                "❗ 절대 금지:\n"
                                "- '전체 키워드는 ~입니다' 문장 생성\n"
                                "- 대표 키워드 선언\n"
                                "- 텍스트에 없는 개념 추가\n"
                                "❗ 반드시 지킬 것:\n"
                                "- 자연스러운 3문장 요약만 생성"
                            ),
                            (
                                "human",
                                "키워드: {keyword}\n\n연도별 내용:\n{summary}"
                            ),
                        ]
                    )

                    combined = "\n".join(f"[{yr}] {txt}" for yr, txt in yearly_summary.items())
                    chain = synthesis_prompt | llm | StrOutputParser()
                    synthesis = chain.invoke({"keyword": keyword, "summary": combined})

                st.subheader(f"키워드 타임라인 : **{keyword}**")

                for yr in [2021, 2022, 2023, 2024, 2025]:
                    st.write(f"### 📌 {yr}년")
                    st.write(yearly_summary[yr])
                    st.markdown("---")

                st.write("### 전체 흐름 요약")
                st.write(synthesis)


    # ---------------------------------------------------
    # 📌 서브탭 3 — 키워드 × 챕터 매핑
    # ---------------------------------------------------
    with sub3:
        st.subheader("Keyword Mapping Table")

        keyword_map = st.text_input(
            "키워드 입력 (예: AI, resale, sustainability, Gen Z, silver spenders...)", key="mapping_keyword"
        )

        if st.button("매핑 생성", key="mapping_button"):
            if not keyword_map.strip():
                st.warning("키워드를 입력해주세요.")
            else:
                import pandas as pd

                rows = []

                with st.spinner("매핑 테이블 생성 중..."):
                    for ch in CHAPTER_LABELS:
                        grouped = search_keyword_timeline(keyword_map, retriever, chapter=ch)

                        # 📌 챕터 내 검색결과 없을 경우
                        if not grouped:
                            rows.append({"Chapter": ch, "Perspective": "관련된 내용이 부족합니다."})
                            continue

                        # 연도별 요약
                        yearly = summarize_yearly_insights(grouped, keyword_map, chapter=ch)

                        # 연도별 텍스트 조합
                        combined = "\n\n".join(
                            f"[{y}]\n{txt}" for y, txt in sorted(yearly.items())
                        )

                        # 📌 핵심 문장 3문장만 생성하도록 제한하는 프롬프트
                        map_prompt = ChatPromptTemplate.from_messages(
                            [
                                (
                                    "system",
                                    "You are a fashion strategy analyst."
                                    "아래 요약 텍스트를 기반으로 해당 챕터가 이 키워드를 어떻게 다루는지 핵심 3문장으로만 정리해줘\n"
                                    "⚠️ 절대 금지:\n"
                                    "- '키워드: ~' 형식 문장 생성 금지\n"
                                    "- '202X년 ~ 흐름은 다음과 같습니다' 금지\n"
                                    "- 텍스트에 없는 숫자/사실/키워드 생성 금지\n"
                                    "⚠️ 반드시 지킬 것:\n"
                                    "- 텍스트 기반 핵심 내용을 자연스러운 3문장으로만 요약\n"
                                    "- 한국어로 서술, 필요한 경우 핵심 용어만 영어 병기"
                                ),
                                (
                                    "human",
                                    "키워드: {keyword}\n챕터: {chapter}\n\n"
                                    "요약 텍스트:\n{summary}"
                                ),
                            ]
                        )

                        chain = map_prompt | llm | StrOutputParser()

                        perspective = chain.invoke(
                            {
                                "keyword": keyword_map,
                                "chapter": ch,
                                "summary": combined,
                            }
                        )

                        rows.append({"Chapter": ch, "Perspective": perspective})

                df = pd.DataFrame(rows)
                st.table(df)

# =====================================================================
# 📌 TAB 2 — 국가별 인사이트
# =====================================================================
with tab_country:

    st.subheader("🌍 Regional Market Insights (2024 & 2025)")

    country = st.selectbox(
        "국가 선택",
        ["🇯🇵 Japan", "🇮🇳 India", "🇺🇸 US", "🇨🇳 China", "🇪🇺 EU"],
        index=0,
    )

    # 국가명을 AI가 이해할 수 있는 텍스트로 변환
    country_map = {
        "🇯🇵 Japan": "Japan",
        "🇮🇳 India": "India",
        "🇺🇸 US": "United States",
        "🇨🇳 China": "China",
        "🇪🇺 EU": "European Union",
    }
    country_text = country_map[country]

    if st.button("국가별 인사이트 생성", key="country_insight"):
        with st.spinner("국가별 시장 인사이트 분석 중..."):

            # 1) RAG 검색: 국가 관련 문서 필터링
            query = f"{country_text} market consumer trend economy fashion"

            docs = vectorstore.similarity_search(query, k=25)

            # 연도별 분리
            docs_2025 = [d.page_content for d in docs if d.metadata.get("year") == 2025]
            docs_2024 = [d.page_content for d in docs if d.metadata.get("year") == 2024]

            def get_summary(texts, year):
                """LLM을 이용한 연도별 요약 함수"""
                if not texts:
                    return f"⚠️ {year}년에는 해당 국가 관련 정보가 거의 없습니다."

                combined = "\n\n".join(texts[:8])  # 너무 긴 경우 압축

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a senior global fashion strategist.\n"
                            "아래 텍스트를 기반으로 해당 국가의 시장 특성을 정확하게 3문장으로만 요약하라.\n\n"
                            "⚠️ 절대 금지:\n"
                            "- '해당 국가의 시장 특성은 다음과 같다' 같은 서론 문장 생성 금지\n"
                            "- 키워드 선언(예: '2025년의 키워드는 ~이다') 금지\n\n"
                            "- '키워드: ~' 형식 금지\n"
                            "- '202X년의 트렌드는 ~입니다' 금지\n"
                            "- '~의 시장 특성은 다음과 같다.' 금지\n"
                            "- '~의 시장은 다음과 같다.' 금지\n"
                            "- 외래 문자·비자연스러운 어구 생성 금지\n"
                            "- 텍스트에 없는 추론/가정/숫자 생성 금지\n"
                            "- 서론·결론·장식적 문장 금지\n\n"
                            "- 결론·조언 문장 금지\n"
                            "⚠️ 반드시 지킬 것:\n"
                            "- 텍스트 기반 핵심만 3문장\n"
                            "- 한국어로 생성, 필요 시 핵심 용어만 영어 병기"
                            "- 오직 텍스트에 있는 사실만 3개의 자연스러운 한국어 문장으로 정리\n"
                            "- 전문적인 문체 유지, 단문/군더더기 없는 표현\n"
                            "- 필요한 경우에만 핵심 용어 영어 병기"
                        ),
                        (
                            "human",
                            f"{year}년의 '{country_text}' 관련 텍스트:\n\n{combined}"
                        ),
                    ]
                )

                chain = prompt | llm | StrOutputParser()
                return chain.invoke({})

            summary_2025 = get_summary(docs_2025, 2025)
            summary_2024 = get_summary(docs_2024, 2024)

        # 출력 UI
        st.markdown(f"### 🌍 {country_text} — Market Insights")

        st.write("### 📌 2025년")
        st.write(summary_2025)
        st.markdown("---")

        st.write("### 📌 2024년")
        st.write(summary_2024)

# =====================================================================
# 📌 TAB — 키워드 시각화 (Top 10 Bar + Top3 Line Chart)
# =====================================================================
with tab_keyword:

    st.subheader("Top 10 Keywords")

    import re
    from collections import Counter
    import pandas as pd
    import plotly.express as px

    # ---------------------------
    # (A) 강화된 키워드 필터링 함수
    # ---------------------------
    def extract_keywords(text):
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", text)
        tokens = [t.lower() for t in tokens if len(t) > 3]

        stopwords = {
            # 일반 영어 불용어
            "that","with","this","have","from","will","into","been","more","than",
            "their","which","also","about","what","when","were","your","them","they",
            "over","only","some","make","made","like","just","very","those","while",
            "where","such","many","each","most","much","other","would","should",
            "could","might","these","both","through","across","there","after","before",
            "under","between","because","based","during","within","without","using",
            "over","well","however","even","though","still","every","including",

            # 숫자 표현
            "percent","million","billion","thousand",

            # 패션 문서에서 너무 기본적인 단어들
            "brands","brand","business","market","industry","consumer","consumers","customer",
            "customers","global","fashion","system","trend","analysis","report",
            "state","chapter","growth","people","products","product","value",
            "goods","retail","sales","year","years","company","companies",

            # 불필요 토큰
            "said","https","http","mckinsey",
        }

        tokens = [t for t in tokens if t not in stopwords]

        # 추가 필터링
        tokens = [t for t in tokens if not t.endswith("ing")]     # 동명사 제거
        tokens = [t for t in tokens if len(set(t)) > 2]           # 반복 문자 제거

        return tokens

    # ---------------------------
    # (B) 연도별 텍스트 취합
    # ---------------------------
    year_texts = {year: "" for year in [2021, 2022, 2023, 2024, 2025]}
    all_docs = list(vectorstore.docstore._dict.values())

    for d in all_docs:
        y = d.metadata.get("year")
        if y in year_texts:
            year_texts[y] += " " + d.page_content

    yearly_keyword_counts = {
        year: Counter(extract_keywords(text))
        for year, text in year_texts.items()
    }

    # ---------------------------
    # (C) 연도 선택 UI
    # ---------------------------
    selected_year = st.selectbox(
        "연도 선택",
        [2021, 2022, 2023, 2024, 2025],
        key="keyword_visual_year"
    )

    st.markdown("---")

    # ---------------------------
    # (D) Bar Chart 출력
    # ---------------------------

    top_keywords = yearly_keyword_counts[selected_year].most_common(10)

    if not top_keywords:
        st.warning("해당 연도에서 의미 있는 키워드를 찾지 못했습니다.")
        st.stop()

    df_bar = pd.DataFrame({
        "keyword": [k for k, _ in top_keywords],
        "count": [v for _, v in top_keywords],
    })

    fig = px.bar(
        df_bar,
        x="keyword",
        y="count",
        title=f"{selected_year} Keyword Top 10",
        color="count",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.write("Top 3 Keywords — Yearly Trend (2021–2025)")

    # ---------------------------
    # (E) 상위 3개 키워드 선택
    # ---------------------------
    top3_keywords = [k for k, _ in top_keywords[:3]]

    # ---------------------------
    # (F) Top3 키워드를 연도별로 빈도 기반 변화 계산
    # ---------------------------
    for keyword in top3_keywords:
        trend_counts = []
        for yr in [2021, 2022, 2023, 2024, 2025]:
            cnt = yearly_keyword_counts[yr][keyword]
            trend_counts.append(cnt)

        df_line = pd.DataFrame({
            "year": ["2021", "2022", "2023", "2024", "2025"],
            "count": trend_counts
        })

        df_line["year"] = df_line["year"].astype(str)

        st.write(f"🔎 {keyword}")

        fig_line = px.line(
            df_line,
            x="year",
            y="count",
            markers=True
        )

        fig_line.update_xaxes(type="category")
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("---")

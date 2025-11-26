import os
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    groq_api_key=groq_key,
)


def extract_year_from_source(source_path: str):
    """
    source 예: 'data/sof25.pdf'
    -> 파일명에서 2025 같은 연도 추출
    """
    import re
    filename = os.path.basename(source_path)
    digits = re.findall(r"\d+", filename)
    if not digits:
        return None
    num = digits[-1]
    if len(num) == 2:
        return 2000 + int(num)
    if len(num) == 4:
        return int(num)
    return None


def search_keyword_timeline(keyword, retriever, chapter: str | None = None):
    """
    keyword -> 연도별 문서 그룹화
    chapter가 지정되면 해당 챕터만 필터링
    """
    query = f"{keyword} 관련된 주요 문장을 찾아줘"
    docs = retriever.invoke(query)

    grouped = defaultdict(list)
    for d in docs:
        # 챕터 필터
        if chapter and d.metadata.get("chapter") != chapter:
            continue

        year = d.metadata.get("year") or extract_year_from_source(
            d.metadata.get("source", "")
        )
        if year:
            grouped[year].append(d.page_content)

    return grouped


def summarize_yearly_insights(grouped_docs, keyword, chapter: str | None = None):
    """
    grouped_docs: {2021: [텍스트들], 2022: [...], ...}
    """
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a fashion trend analyst. "
                "각 연도별 문단을 읽고, 해당 키워드의 변화 흐름을 분석해줘. "
                "답변은 한국어로, 핵심 용어는 영어 병기해줘.",
            ),
            (
                "human",
                "키워드: {keyword}\n"
                "챕터: {chapter}\n\n"
                "{year}년 문단들:\n{docs}\n\n"
                "➡ 이 연도의 핵심 인사이트를 3문장으로 요약해줘.",
            ),
        ]
    )

    yearly_summary = {}
    ch_label = chapter or "전체"

    for year, docs in sorted(grouped_docs.items()):
        combined = "\n\n".join(docs[:3])  # 연도별 최대 3문단만 사용

        chain = summary_prompt | llm | StrOutputParser()
        summary = chain.invoke(
            {
                "keyword": keyword,
                "year": year,
                "chapter": ch_label,
                "docs": combined,
            }
        )

        yearly_summary[year] = summary

    return yearly_summary


def generate_timeline_synthesis(yearly_summary, keyword, chapter: str | None = None):
    """
    2021~2025 전체 흐름 요약 생성
    """
    synthesis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior fashion strategist. "
                "연도별 주요 변화를 종합해 '트렌드 타임라인'을 만들어줘.",
            ),
            (
                "human",
                "키워드: {keyword}\n"
                "챕터: {chapter}\n\n"
                "연도별 정리:\n{summary}\n\n"
                "➡ 이 전체 흐름을 한 문단으로 설명해줘.",
            ),
        ]
    )

    combined = ""
    for year, text in yearly_summary.items():
        combined += f"\n[{year}]\n{text}\n"

    ch_label = chapter or "전체"

    chain = synthesis_prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "keyword": keyword,
            "chapter": ch_label,
            "summary": combined,
        }
    )

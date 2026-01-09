import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import streamlit as st

from src.api import chat_completion

# 向量检索模块（可选导入）
try:
    from src.data_process import RAGPipeline
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False

# 知识库目录（优先清洗后的）
KB_DIRS = [
    Path("data/output/民法典/articles_clean"),
    Path("data/articles"),
]


def parse_kb_file(path: Path) -> Optional[Dict[str, str]]:
    """Parse a knowledge file into heading/title/body."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    heading, title, body = "", "", text
    parts = text.split(maxsplit=2)
    if len(parts) >= 3:
        heading, title, body = parts
    elif len(parts) == 2:
        heading, title = parts
        body = ""
    else:
        # fallback: try filename
        stem = path.stem
        if "_" in stem:
            heading, title = stem.split("_", 1)
        else:
            heading = stem
        body = text
    return {"heading": heading, "title": title, "body": body, "source": str(path)}


@st.cache_data(show_spinner=False)
def load_kb() -> Tuple[List[Dict[str, str]], Optional[Path]]:
    """Load knowledge base from first existing dir."""
    for kb_dir in KB_DIRS:
        if not kb_dir.exists():
            continue
        entries: List[Dict[str, str]] = []
        for path in sorted(kb_dir.glob("*.txt")):
            parsed = parse_kb_file(path)
            if parsed:
                entries.append(parsed)
        if entries:
            return entries, kb_dir
    return [], None


def extract_keywords(question: str, model: str = "deepseek-chat") -> List[str]:
    """Ask LLM to extract 2-5 keywords (逗号分隔)。"""
    prompt = (
        "请从下列问题中提取2-5个中文关键词，使用逗号分隔，只输出关键词（我希望这个关键词专业一点，便于检索具体的法律条文）：\n"
        f"{question}"
    )
    resp = chat_completion(
        [
            {"role": "system", "content": "你是简洁的关键词提取器，只输出关键词。"},
            {"role": "user", "content": prompt},
        ],
        model=model,
    )
    parts = re.split(r"[，,\n\s]+", resp)
    return [p.strip() for p in parts if p.strip()]


def extract_question_tokens(question: str) -> List[str]:
    """
    从原问题中提取2-3字中文片段，作为检索补充，避免仅依赖模型提取。
    """
    tokens: Set[str] = set()
    text = "".join(re.findall(r"[\u4e00-\u9fff]", question))
    for size in (2, 3):
        for i in range(len(text) - size + 1):
            tokens.add(text[i : i + size])
    tokens = {t for t in tokens if len(set(t)) > 1}
    return list(tokens)


def search_kb(
    keywords: List[str],
    question_tokens: List[str],
    entries: List[Dict[str, str]],
    top_k: int = 5,
) -> List[Dict[str, str]]:
    """Simple scoring by keyword/question-token occurrence."""
    weight_kw = 3
    weight_qt = 1
    scored: List[Tuple[int, Dict[str, str]]] = []
    all_tokens = [(k, weight_kw) for k in keywords] + [
        (t, weight_qt) for t in question_tokens
    ]
    for e in entries:
        haystack = f"{e['heading']} {e['title']} {e['body']}"
        score = 0
        for token, w in all_tokens:
            if not token:
                continue
            score += w * haystack.count(token)
        if score > 0:
            scored.append((score, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return entries[:top_k]
    return [e for _, e in scored[:top_k]]


def generate_answer(
    question: str,
    context_entries: List[Dict[str, str]],
    model: str = "deepseek-chat",
) -> str:
    context_text = "\n".join(
        f"{e['heading']} {e['title']}: {e['body']}" for e in context_entries
    )
    prompt = f"""你是法律助手，请严格依赖背景知识回答。
背景知识：
{context_text}

用户问题：
{question}

要求：先引用背景中的关键句作简洁回答；如果背景不足以回答，请明确说明“不足以依据背景知识回答”。"""
    return chat_completion(
        [
            {"role": "system", "content": "你是严谨的法律问答助手，优先使用提供的背景知识。"},
            {"role": "user", "content": prompt},
        ],
        model=model,
    )


@st.cache_resource
def get_rag_pipeline():
    """初始化并缓存 RAG Pipeline（向量检索）"""
    if not VECTOR_SEARCH_AVAILABLE:
        return None
    pipeline = RAGPipeline()
    return pipeline


def main() -> None:
    st.title("SmartTool RAG 法律问答系统")
    
    # 检索模式选择
    search_modes = ["关键词匹配"]
    if VECTOR_SEARCH_AVAILABLE:
        search_modes.append("向量语义检索")
    
    search_mode = st.radio(
        "检索模式",
        search_modes,
        horizontal=True,
        help="向量语义检索需要先构建索引，效果更好但首次加载较慢",
    )
    
    use_vector_search = search_mode == "向量语义检索"
    
    # 根据检索模式加载知识库
    if use_vector_search:
        pipeline = get_rag_pipeline()
        if pipeline is None:
            st.error("向量检索模块未安装，请运行: pip install chromadb sentence-transformers")
            return
        
        if not pipeline.is_ready:
            st.warning("向量索引未构建，正在构建索引（首次需要几分钟）...")
            with st.spinner("构建向量索引中..."):
                count = pipeline.build_index(show_progress=False)
            st.success(f"索引构建完成，文档数: {count}")
        else:
            st.success(f"向量索引已加载，文档数: {pipeline.document_count}")
        
        kb_entries = None  # 向量检索不需要加载全部条目
    else:
        st.write("步骤：提取关键词 -> 本地检索 -> 组装背景 -> 调用 LLM 生成答案。")
        kb_entries, kb_dir = load_kb()
        if not kb_entries:
            st.error("未找到知识库，请先准备数据目录：data/output/民法典/articles_clean 或 data/articles")
            return
        else:
            st.success(f"知识库已加载，条目数：{len(kb_entries)}，来源：{kb_dir}")

    user_question = st.text_area("请输入你的问题", height=150)
    model = st.text_input("模型名称", value="deepseek-chat")
    top_k = st.slider("召回条目数", min_value=3, max_value=15, value=8, step=1)

    if st.button("开始检索并生成"):
        if not user_question.strip():
            st.warning("问题不能为空")
            return
        
        if use_vector_search:
            # 向量语义检索
            with st.spinner("语义检索中..."):
                hits = pipeline.search_formatted(user_question, top_k=top_k)
            st.write("命中条目（按语义相似度排序）：")
            for e in hits:
                score = e.get("score", 0)
                st.markdown(f"- **{e['heading']} {e['title']}** (相似度: {score:.2%})：{e['body'][:200]}...")
        else:
            # 关键词检索
            with st.spinner("提取关键词..."):
                keywords = extract_keywords(user_question, model=model)
            question_tokens = extract_question_tokens(user_question)
            st.write(f"关键词：{', '.join(keywords) if keywords else '（无）'}")

            with st.spinner("检索知识库..."):
                hits = search_kb(keywords, question_tokens, kb_entries, top_k=top_k)
            st.write("命中条目：")
            for e in hits:
                st.markdown(f"- **{e['heading']} {e['title']}**：{e['body']}")

        with st.spinner("生成回答..."):
            answer = generate_answer(user_question, hits, model=model)
        st.success("生成完成")
        st.markdown("### 回答")
        st.markdown(answer)


if __name__ == "__main__":
    main()

import argparse
import concurrent.futures as cf
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from src.api import chat_completion
    from src.getlaws import parse_articles, save_articles
    from src.pdf2text import read_pdf_text
except ImportError:
    # Fallback when running as a script without package context
    from api import chat_completion  # type: ignore
    from getlaws import parse_articles, save_articles  # type: ignore
    from pdf2text import read_pdf_text  # type: ignore

RawArticle = Tuple[int, str, str]  # number, heading_cn (e.g., 第一条), raw_body
CleanedArticle = Tuple[int, str, str, str]  # number, heading_cn, title, cleaned_body

# Allowed Chinese punctuation (extend if needed)
ALLOWED_PUNCT = "，。、；：？！、（）《》“”‘’—…·"  # 去掉【】以便被清理
CHINESE_AND_PUNCT_PATTERN = re.compile(rf"[^\u4e00-\u9fff{ALLOWED_PUNCT}\s]")
HEADING_CN_PATTERN = re.compile(r"^第[\d一二三四五六七八九十百千万〇零○两]+条$")

# Title extraction helpers (for fallback /补全)
BRACKET_TITLE_PATTERN = re.compile(r"^\s*[【\[]\s*([^】\]]{1,40}?)\s*[】\]]")
PAREN_TITLE_PATTERN = re.compile(r"^\s*[（(]\s*([^)）]{1,40}?)\s*[)）]")

# System prompt
SYSTEM_PROMPT = (
    "你是法律条文清洗助手。"
    "目标：接收多条中国民法典条文，删除脚注/序号/星号等杂项标记（如[①][1]①*等），"
    "移除所有非中文且非中文标点的字符，合并多余空格，保持原句序和含义，不意译、不增删要点。"
    "输出必须严格为 JSON 数组，不要有任何额外文字。数组内每个元素是对象，字段固定为："
    "`{\"heading_cn\":\"第X条\",\"title\":\"标题（不含编号/括号）\",\"body\":\"正文（不含括号）\"}`。"
    "heading_cn 必须与输入保持一致（如“第十条”），只出现一次；title 必须是条文标题，不能只写数字或条号，不能包含条文号或括号；body 不含括号、无换行。"
    "数组按条文号升序。不要返回 TSV、不要返回解释、不要在 JSON 外加文本。"
)


def strip_non_chinese_and_punct(text: str) -> str:
    """Remove all chars that are not Chinese or allowed punctuation, then squish spaces."""
    cleaned = CHINESE_AND_PUNCT_PATTERN.sub("", text)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def strip_heading_from_body(heading_cn: str, body: str) -> str:
    """
    Remove leading heading like '第十条'及其后的括号/冒号/空白，避免重复编号出现在正文。
    """
    pattern = re.compile(rf"^{re.escape(heading_cn)}[：:【\s]*")
    return pattern.sub("", body, count=1)


def extract_title_and_body_from_raw(heading_cn: str, raw_body: str) -> Tuple[str, str]:
    """
    Best-effort local extraction:
    - remove leading heading
    - if body begins with 【标题】 or (标题)/(标题) etc., extract as title
    - remove the extracted title marker from body
    Returns (title, body_without_title_marker)
    """
    body = strip_heading_from_body(heading_cn, raw_body).strip()

    # Try 【...】 or [...] title
    m = BRACKET_TITLE_PATTERN.search(body)
    if m:
        title = m.group(1).strip()
        body = body[m.end() :].lstrip(" ：:，。；;")
        return title, body

    # Try （...） or (...) title
    m = PAREN_TITLE_PATTERN.search(body)
    if m:
        title = m.group(1).strip()
        body = body[m.end() :].lstrip(" ：:，。；;")
        return title, body

    return "", body


def build_batch_prompt(batch: Sequence[RawArticle]) -> str:
    items = []
    for number, heading, body in batch:
        body_wo_heading = strip_heading_from_body(heading, body)
        items.append(
            {
                "heading_cn": heading,
                "title": "",
                "body": " ".join(body_wo_heading.split()),
            }
        )
    return "请按要求返回 JSON 数组。输入条文列表：" + json.dumps(items, ensure_ascii=False)


def parse_cleaned_response(
    response: str, fallback_batch: Sequence[RawArticle]
) -> List[CleanedArticle]:
    """
    Parse JSON response. If parsing fails, return empty list to trigger retry/fallback.
    """
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return []

    cleaned_map: Dict[int, Tuple[str, str]] = {}
    heading_to_num = {h: n for n, h, _ in fallback_batch}

    if not isinstance(data, list):
        return []

    for item in data:
        if not isinstance(item, dict):
            continue
        heading_raw = item.get("heading_cn", "")
        title_raw = item.get("title", "")
        body_raw = item.get("body", "")
        if heading_raw not in heading_to_num:
            continue

        num = heading_to_num[heading_raw]
        heading = heading_raw  # keep original heading

        title = strip_non_chinese_and_punct(str(title_raw))
        body = strip_non_chinese_and_punct(str(body_raw))

        # Ensure no repeated heading in title/body
        title = strip_heading_from_body(heading, title)
        body = strip_heading_from_body(heading, body)

        # If title is empty, try to infer from the raw article (e.g., 【标题】)
        if not title:
            raw_body = next((b for n, h, b in fallback_batch if n == num and h == heading), "")
            inferred_title, _ = extract_title_and_body_from_raw(heading, raw_body)
            title = strip_non_chinese_and_punct(inferred_title)

        cleaned_map[num] = (title, body)

    cleaned_list: List[CleanedArticle] = []
    for num, heading_cn, raw_body in fallback_batch:
        if num in cleaned_map:
            fixed_title, cleaned_body = cleaned_map[num]
            cleaned_list.append((num, heading_cn, fixed_title, cleaned_body))
        else:
            # Last resort: local extract + clean
            inferred_title, body_wo_title = extract_title_and_body_from_raw(heading_cn, raw_body)
            cleaned_list.append(
                (
                    num,
                    heading_cn,
                    strip_non_chinese_and_punct(inferred_title),
                    strip_non_chinese_and_punct(" ".join(body_wo_title.split())),
                )
            )
    return cleaned_list


def local_clean_fallback(batch: Sequence[RawArticle]) -> List[CleanedArticle]:
    """
    A local deterministic cleaner to avoid AI when outputs are invalid:
    - Title: try extract from 【标题】/（标题）; if not found, keep empty (still saved, but may violate your validator).
    - Body: strip non-Chinese/punct, collapse spaces.
    """
    results: List[CleanedArticle] = []
    for num, heading_cn, raw_body in batch:
        inferred_title, body_wo_title = extract_title_and_body_from_raw(heading_cn, raw_body)
        title = strip_non_chinese_and_punct(inferred_title)
        cleaned_body = strip_non_chinese_and_punct(" ".join(body_wo_title.split()))
        results.append((num, heading_cn, title, cleaned_body))
    return results


def validate_cleaned_batch(
    cleaned: Sequence[CleanedArticle], fallback_batch: Sequence[RawArticle]
) -> List[str]:
    errors: List[str] = []
    numeric_title_pattern = re.compile(r"^[0-9一二三四五六七八九十百千万〇零○两\s]+$")

    if len(cleaned) != len(fallback_batch):
        errors.append(f"条数不符: 期望 {len(fallback_batch)} 得到 {len(cleaned)}")
        return errors

    for (num, heading_cn, title, body), raw in zip(cleaned, fallback_batch):
        raw_num, raw_heading, raw_body = raw

        if num != raw_num:
            errors.append(f"编号不匹配: 期望{raw_num} 实际{num}")
        if heading_cn != raw_heading:
            errors.append(f"条文号变动: 期望{raw_heading} 得到{heading_cn}")
        if not HEADING_CN_PATTERN.match(heading_cn):
            errors.append(f"条文号格式错误: {heading_cn}")

        if not title:
            errors.append(f"标题为空: {heading_cn}")
        if HEADING_CN_PATTERN.search(title):
            errors.append(f"标题包含条文号: {heading_cn} -> {title}")
        if "第" in title and "条" in title:
            errors.append(f"标题疑似重复编号: {heading_cn} -> {title}")
        if numeric_title_pattern.fullmatch(title):
            errors.append(f"标题疑似仅为数字/编号: {heading_cn} -> {title}")

        stripped_heading = heading_cn.replace("第", "").replace("条", "")
        if title == heading_cn or title == stripped_heading:
            errors.append(f"标题与条号相同或仅去除“第/条”: {heading_cn} -> {title}")

        if "\t" in title or "\n" in title or "【" in title or "】" in title or "[" in title or "]" in title:
            errors.append(f"标题含非法字符/分隔符: {heading_cn} -> {title}")

        if not body:
            errors.append(f"正文为空: {heading_cn}")
        if "\t" in body or "\n" in body or "【" in body or "】" in body or "[" in body or "]" in body:
            errors.append(f"正文含非法字符/分隔符: {heading_cn}")

        # 关键修复：不要禁止正文出现“第…条”的引用，只检查正文开头是否重复出现本条 heading
        if re.match(rf"^\s*{re.escape(heading_cn)}", body):
            errors.append(f"正文开头重复条文号: {heading_cn}")

    return errors


def clean_batch(
    batch: Sequence[RawArticle],
    model: str = "deepseek-chat",
) -> List[CleanedArticle]:
    prompt = build_batch_prompt(batch)
    attempts = 10
    last_cleaned: List[CleanedArticle] = []

    for attempt in range(attempts):
        user_content = (
            prompt
            if attempt == 0
            else prompt
            + "\n上次输出不是合法 JSON。请严格只输出 JSON 数组本身，不要任何解释，不要 Markdown/代码块，不要 TSV。"
        )

        response = chat_completion(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            model=model,
        )

        last_cleaned = parse_cleaned_response(response, batch)
        errors = validate_cleaned_batch(last_cleaned, batch)
        if not errors:
            return last_cleaned

    # If still invalid, fallback to local cleaned from raw (deterministic)
    return local_clean_fallback(batch)


def chunk_articles(
    articles: Sequence[RawArticle], batch_size: int
) -> List[List[RawArticle]]:
    return [list(articles[i : i + batch_size]) for i in range(0, len(articles), batch_size)]


def clean_articles(
    articles: Sequence[RawArticle],
    batch_size: int = 8,
    max_workers: int = 4,
    model: str = "deepseek-chat",
    progress: bool = True,
) -> List[CleanedArticle]:
    """
    Clean articles in batches with optional multithreading.
    """
    batches = chunk_articles(articles, batch_size)
    cleaned: List[CleanedArticle] = []

    total_batches = len(batches)
    if progress:
        print(f"开始清洗: {len(articles)} 条, 批大小 {batch_size}, 批次数 {total_batches}")

    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(clean_batch, batch, model): batch for batch in batches}
        done = 0
        for future in cf.as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                cleaned.extend(future.result())
            except Exception:
                # last resort fallback
                cleaned.extend(local_clean_fallback(batch))
            finally:
                done += 1
                if progress:
                    print(f"  清洗进度: {done}/{total_batches} 批", end="\r", flush=True)

    if progress:
        print(f"\n清洗完成，共 {len(cleaned)} 条。")

    cleaned.sort(key=lambda x: x[0])
    return cleaned


def save_cleaned_articles(articles: Iterable[CleanedArticle], output_dir: str) -> None:
    """
    Save cleaned articles to files, each with fixed single-line content:
    `<第X条> <标题> <正文>`
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for number, heading_cn, title, body in articles:
        safe_title = title or "无标题"
        filename = f"{number:04d}_{safe_title}.txt"
        line = f"{heading_cn} {safe_title} {body}".strip() + "\n"
        (output_path / filename).write_text(line, encoding="utf-8")


def filter_articles(
    articles: Sequence[RawArticle],
    start_article: Optional[int],
    end_article: Optional[int],
) -> List[RawArticle]:
    return [
        a
        for a in articles
        if (start_article is None or a[0] >= start_article)
        and (end_article is None or a[0] <= end_article)
    ]


def process_pdf_to_cleaned(
    pdf_path: str,
    work_root: str = "data/output",
    batch_size: int = 8,
    max_workers: int = 4,
    model: str = "deepseek-chat",
    start_article: Optional[int] = None,
    end_article: Optional[int] = None,
    progress: bool = True,
) -> Path:
    """
    Pipeline:
    1) PDF -> raw TXT
    2) TXT -> 分条 (原文)
    3) 原文批量送 AI 清洗
    输出目录名称使用源 pdf/txt 的文件名。
    """
    pdf_file = Path(pdf_path)
    base_name = pdf_file.stem
    base_dir = Path(work_root) / base_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: PDF -> raw txt
    raw_txt_path = base_dir / f"{base_name}.txt"
    raw_text = read_pdf_text(pdf_file)
    raw_txt_path.write_text(raw_text, encoding="utf-8")
    if progress:
        print(f"已提取 PDF 文本 -> {raw_txt_path}")

    # Step 2: split into articles (raw) and save
    articles = parse_articles(raw_text)
    articles = filter_articles(articles, start_article, end_article)
    raw_articles_dir = base_dir / "articles_raw"
    save_articles(articles, raw_articles_dir)
    if progress:
        print(f"已切分条文 {len(articles)} 条 -> {raw_articles_dir}")

    # Step 3: clean via AI in batches
    cleaned_articles = clean_articles(
        articles,
        batch_size=batch_size,
        max_workers=max_workers,
        model=model,
        progress=progress,
    )
    cleaned_dir = base_dir / "articles_clean"
    save_cleaned_articles(cleaned_articles, str(cleaned_dir))
    if progress:
        print(f"已保存清洗结果 -> {cleaned_dir}")

    return base_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF->TXT->条文切分->AI清洗 的批处理工具")
    parser.add_argument("--pdf", required=True, help="源 PDF 路径")
    parser.add_argument(
        "--work-root",
        default="data/output",
        help="输出根目录（会在其中以 pdf 名称创建子目录）",
    )
    parser.add_argument("--start", type=int, default=None, help="起始条文号（包含）")
    parser.add_argument("--end", type=int, default=None, help="结束条文号（包含）")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="每次送入 AI 的条文数量（越大越省钱，但不要太大以免超长）",
    )
    parser.add_argument("--max-workers", type=int, default=64, help="并发批次数（线程数）")
    parser.add_argument("--model", default="deepseek-chat", help="聊天模型名称")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度输出")

    args = parser.parse_args()
    base_dir = process_pdf_to_cleaned(
        pdf_path=args.pdf,
        work_root=args.work_root,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        model=args.model,
        start_article=args.start,
        end_article=args.end,
        progress=not args.no_progress,
    )
    print(f"完成。输出目录：{base_dir.resolve()}")


if __name__ == "__main__":
    main()

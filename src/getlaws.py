import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Matches real article headings (start of line), e.g., "第十二条" or "第 12 条".
# Uses a lookahead for a heading bracket to avoid inline references like
# "第五百一十条的规定".
ARTICLE_PATTERN = re.compile(
    r"(?m)^[ \t]*(第\s*[一二三四五六七八九十百千万〇零○\d]+\s*条)(?=\s*(?:[（(【]|$))"
)


def chinese_numeral_to_int(value: str) -> int:
    """
    Convert a Chinese article number (e.g., '第十二条' or '第12条') to an int.
    Supports numbers up to the thousands, which covers the Civil Code.
    """
    cleaned = re.sub(r"[第条\\s]", "", value)

    # Fast path for Arabic digits
    digits_only = "".join(ch for ch in cleaned if ch.isdigit())
    if digits_only:
        return int(digits_only)

    digit_map = {
        "零": 0,
        "〇": 0,
        "○": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    unit_map = {"十": 10, "百": 100, "千": 1000, "万": 10000}

    total = 0
    section = 0
    number = 0

    for ch in cleaned:
        if ch in digit_map:
            number = digit_map[ch]
        elif ch in unit_map:
            unit = unit_map[ch]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
                number = 0
            else:
                if number == 0 and unit == 10:
                    # "十" at the start means 10, "十五" means 15, etc.
                    number = 1
                section += number * unit
                number = 0

    return total + section + number


def parse_articles(raw_text: str) -> List[Tuple[int, str, str]]:
    """
    Split the Civil Code text into articles.
    Returns a list of (article_number, article_heading, article_body).
    """
    normalized = (
        raw_text.replace("\ufeff", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\f", "\n")
    )

    matches = list(ARTICLE_PATTERN.finditer(normalized))
    articles: List[Tuple[int, str, str]] = []

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        heading_raw = match.group(1)
        # Remove any whitespace/newlines so filenames stay valid
        heading = re.sub(r"\s+", "", heading_raw)
        number = chinese_numeral_to_int(heading)
        body = normalized[start:end].strip()
        articles.append((number, heading, body))

    return articles


def save_articles(
    articles: Iterable[Tuple[int, str, str]], output_dir: str
) -> List[Path]:
    """
    Save each article to an individual UTF-8 text file.
    Filenames are zero-padded so they sort in numeric order.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for number, heading, body in articles:
        filename = f"{number:04d}_{heading}.txt"
        path = output_path / filename
        path.write_text(body, encoding="utf-8")
        saved_paths.append(path)

    return saved_paths


def extract_articles(
    input_file: str,
    output_dir: str,
    start_article: Optional[int] = None,
    end_article: Optional[int] = None,
) -> List[Path]:
    """
    Extract articles from the Civil Code text file and optionally filter by range.
    start_article/end_article are inclusive when provided.
    """
    text = Path(input_file).read_text(encoding="utf-8")
    articles = parse_articles(text)

    if start_article is not None or end_article is not None:
        articles = [
            a
            for a in articles
            if (start_article is None or a[0] >= start_article)
            and (end_article is None or a[0] <= end_article)
        ]

    return save_articles(articles, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split 民法典.txt into per-article files"
    )
    parser.add_argument(
        "--input",
        default=Path("data") / "民法典.txt",
        help="Path to 民法典.txt",
    )
    parser.add_argument(
        "--output",
        default=Path("data") / "articles",
        help="Directory to store extracted articles",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start article number (inclusive). Omit to start from the first article.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End article number (inclusive). Omit to include all remaining articles.",
    )

    args = parser.parse_args()
    saved = extract_articles(
        str(args.input), str(args.output), args.start, args.end
    )
    print(f"Saved {len(saved)} article file(s) to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

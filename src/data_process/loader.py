"""
文档加载器模块

负责从文件系统加载法律条文，解析为结构化 Document 对象。
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Document:
    """文档数据类，存储条文内容和元数据"""
    
    id: str  # 唯一标识，通常为文件名（不含扩展名）
    content: str  # 条文完整文本
    metadata: Dict[str, str] = field(default_factory=dict)
    
    @property
    def heading(self) -> str:
        """条文编号，如 '第一条'"""
        return self.metadata.get("heading", "")
    
    @property
    def title(self) -> str:
        """条文标题"""
        return self.metadata.get("title", "")
    
    @property
    def body(self) -> str:
        """条文正文"""
        return self.metadata.get("body", self.content)
    
    @property
    def source(self) -> str:
        """来源文件路径"""
        return self.metadata.get("source", "")


class DocumentLoader:
    """文档加载器，从目录加载 txt 文件为 Document 对象"""
    
    # 默认知识库目录（优先级从高到低）
    DEFAULT_KB_DIRS = [
        Path("data/output/民法典/articles_clean"),
        Path("data/articles"),
    ]
    
    def __init__(self, kb_dir: Optional[Path] = None):
        """
        初始化加载器
        
        Args:
            kb_dir: 知识库目录路径，若为 None 则自动查找默认目录
        """
        self.kb_dir = kb_dir or self._find_kb_dir()
    
    def _find_kb_dir(self) -> Optional[Path]:
        """查找第一个存在的默认知识库目录"""
        for d in self.DEFAULT_KB_DIRS:
            if d.exists() and any(d.glob("*.txt")):
                return d
        return None
    
    def load(self) -> List[Document]:
        """
        加载知识库目录下所有 txt 文件
        
        Returns:
            Document 对象列表
        """
        if self.kb_dir is None or not self.kb_dir.exists():
            return []
        
        documents = []
        for path in sorted(self.kb_dir.glob("*.txt")):
            doc = self._parse_file(path)
            if doc:
                documents.append(doc)
        
        return documents
    
    def _parse_file(self, path: Path) -> Optional[Document]:
        """
        解析单个文件为 Document 对象
        
        Args:
            path: 文件路径
            
        Returns:
            Document 对象，解析失败返回 None
        """
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception:
            return None
        
        if not text:
            return None
        
        # 解析文本结构：编号 标题 正文
        heading, title, body = "", "", text
        parts = text.split(maxsplit=2)
        
        if len(parts) >= 3:
            heading, title, body = parts
        elif len(parts) == 2:
            heading, title = parts
            body = ""
        else:
            # 尝试从文件名解析
            stem = path.stem
            if "_" in stem:
                heading, title = stem.split("_", 1)
            else:
                heading = stem
            body = text
        
        return Document(
            id=path.stem,
            content=text,
            metadata={
                "heading": heading,
                "title": title,
                "body": body,
                "source": str(path),
            }
        )
    
    def load_from_texts(self, texts: List[str], ids: Optional[List[str]] = None) -> List[Document]:
        """
        从文本列表创建 Document 对象（用于自定义数据）
        
        Args:
            texts: 文本列表
            ids: 可选的 ID 列表，若不提供则自动生成
            
        Returns:
            Document 对象列表
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        return [
            Document(id=doc_id, content=text, metadata={"body": text})
            for doc_id, text in zip(ids, texts)
        ]


if __name__ == "__main__":
    # 测试加载器
    loader = DocumentLoader()
    if loader.kb_dir:
        print(f"知识库目录: {loader.kb_dir}")
        docs = loader.load()
        print(f"加载文档数: {len(docs)}")
        if docs:
            print(f"示例文档: {docs[0].heading} {docs[0].title}")
    else:
        print("未找到知识库目录")


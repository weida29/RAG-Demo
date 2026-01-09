"""
RAG 数据处理 Pipeline

提供一键构建索引和检索的完整流程。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .loader import DocumentLoader, Document
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever


class RAGPipeline:
    """RAG 数据处理 Pipeline，编排完整的索引构建和检索流程"""
    
    def __init__(
        self,
        kb_dir: Optional[Path] = None,
        persist_dir: Optional[Path] = None,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        初始化 Pipeline
        
        Args:
            kb_dir: 知识库目录路径
            persist_dir: 向量存储持久化目录
            model_name: Embedding 模型名称
            collection_name: Chroma 集合名称
        """
        self.loader = DocumentLoader(kb_dir=kb_dir)
        self.embedder = Embedder(model_name=model_name)
        self.vector_store = VectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )
        
        self._documents: List[Document] = []
    
    def build_index(
        self,
        force_rebuild: bool = False,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> int:
        """
        构建向量索引
        
        Args:
            force_rebuild: 是否强制重建（清空现有数据）
            batch_size: 向量化批处理大小
            show_progress: 是否显示进度
            
        Returns:
            索引的文档数量
        """
        # 检查是否需要重建
        if not force_rebuild and self.vector_store.exists():
            print(f"索引已存在，文档数: {self.vector_store.count()}")
            print("使用 force_rebuild=True 强制重建")
            return self.vector_store.count()
        
        # 清空现有数据
        if force_rebuild and self.vector_store.exists():
            print("清空现有索引...")
            self.vector_store.clear()
        
        # 加载文档
        print(f"加载文档: {self.loader.kb_dir}")
        self._documents = self.loader.load()
        
        if not self._documents:
            print("未找到文档，请检查知识库目录")
            return 0
        
        print(f"加载文档数: {len(self._documents)}")
        
        # 向量化
        print("向量化文档...")
        texts = [doc.content for doc in self._documents]
        embeddings = self.embedder.embed_batch(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        
        # 存储到向量数据库
        print("存储到向量数据库...")
        self.vector_store.add_documents(self._documents, embeddings)
        
        count = self.vector_store.count()
        print(f"索引构建完成，文档数: {count}")
        
        return count
    
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        语义检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        return self.retriever.retrieve(query, top_k=top_k)
    
    def search_formatted(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, str]]:
        """
        语义检索，返回格式化结果（兼容 main.py）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            格式化的结果列表
        """
        return self.retriever.retrieve_with_context(query, top_k=top_k)
    
    @property
    def is_ready(self) -> bool:
        """检查 Pipeline 是否就绪"""
        return self.retriever.is_ready
    
    @property
    def document_count(self) -> int:
        """返回索引的文档数量"""
        return self.vector_store.count()
    
    @property
    def kb_dir(self) -> Optional[Path]:
        """返回知识库目录"""
        return self.loader.kb_dir


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG 数据处理 Pipeline")
    parser.add_argument(
        "--kb-dir",
        type=str,
        help="知识库目录路径",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="data/chroma_db",
        help="向量存储目录（默认: data/chroma_db）",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Embedding 模型名称",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制重建索引",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="测试查询",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回结果数量（默认: 5）",
    )
    
    args = parser.parse_args()
    
    # 初始化 Pipeline
    pipeline = RAGPipeline(
        kb_dir=Path(args.kb_dir) if args.kb_dir else None,
        persist_dir=Path(args.persist_dir),
        model_name=args.model,
    )
    
    # 构建索引
    if args.rebuild or not pipeline.is_ready:
        pipeline.build_index(force_rebuild=args.rebuild)
    else:
        print(f"索引已就绪，文档数: {pipeline.document_count}")
    
    # 测试查询
    if args.query:
        print(f"\n查询: {args.query}")
        print("-" * 50)
        results = pipeline.search(args.query, top_k=args.top_k)
        
        for i, r in enumerate(results):
            print(f"\n[{i+1}] 相似度: {r['score']:.4f}")
            metadata = r.get("metadata", {})
            heading = metadata.get("heading", "")
            title = metadata.get("title", "")
            print(f"    {heading} {title}")
            print(f"    {r['content'][:200]}...")


if __name__ == "__main__":
    main()


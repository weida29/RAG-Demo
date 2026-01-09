"""
语义检索器模块

组合 Embedder 和 VectorStore，提供端到端的语义检索功能。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .embedder import Embedder
from .vector_store import VectorStore


class Retriever:
    """语义检索器，基于向量相似度检索相关文档"""
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        persist_dir: Optional[Path] = None,
        model_name: Optional[str] = None,
    ):
        """
        初始化检索器
        
        Args:
            embedder: 向量化器实例，若为 None 则自动创建
            vector_store: 向量存储实例，若为 None 则自动创建
            persist_dir: 向量存储持久化目录
            model_name: Embedding 模型名称
        """
        self.embedder = embedder or Embedder(model_name=model_name)
        self.vector_store = vector_store or VectorStore(persist_dir=persist_dir)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        根据查询文本检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            where: 元数据过滤条件
            
        Returns:
            检索结果列表，每个元素包含：
            - id: 文档 ID
            - content: 文档内容
            - metadata: 元数据（heading, title, body, source）
            - distance: 向量距离（越小越相似）
            - score: 相似度分数（0-1，越大越相似）
        """
        # 向量化查询文本
        query_embedding = self.embedder.embed(query)
        
        # 在向量存储中检索
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )
        
        # 添加相似度分数（将距离转换为分数）
        for r in results:
            # Chroma 使用余弦距离，距离范围 [0, 2]，转换为相似度 [0, 1]
            r["score"] = 1 - r["distance"] / 2
        
        return results
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        context_format: str = "full",
    ) -> List[Dict[str, str]]:
        """
        检索并返回格式化的上下文，兼容 main.py 的格式
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            context_format: 上下文格式
                - "full": 完整内容
                - "structured": 结构化（heading, title, body）
                
        Returns:
            格式化的结果列表，与 main.py 的 search_kb 返回格式兼容
        """
        results = self.retrieve(query, top_k=top_k)
        
        formatted = []
        for r in results:
            metadata = r.get("metadata", {})
            formatted.append({
                "heading": metadata.get("heading", ""),
                "title": metadata.get("title", ""),
                "body": metadata.get("body", r["content"]),
                "source": metadata.get("source", ""),
                "score": r.get("score", 0.0),
            })
        
        return formatted
    
    @property
    def is_ready(self) -> bool:
        """检查检索器是否就绪（向量存储中有数据）"""
        return self.vector_store.exists()
    
    @property
    def document_count(self) -> int:
        """返回索引的文档数量"""
        return self.vector_store.count()


if __name__ == "__main__":
    # 测试检索器（需要先运行 pipeline 构建索引）
    retriever = Retriever()
    
    if retriever.is_ready:
        print(f"检索器就绪，文档数: {retriever.document_count}")
        
        # 测试检索
        query = "什么是民事权利能力"
        results = retriever.retrieve(query, top_k=3)
        
        print(f"\n查询: {query}")
        print(f"结果数: {len(results)}")
        for i, r in enumerate(results):
            print(f"\n[{i+1}] 相似度: {r['score']:.4f}")
            print(f"    ID: {r['id']}")
            print(f"    内容: {r['content'][:100]}...")
    else:
        print("检索器未就绪，请先运行 pipeline.py 构建索引")


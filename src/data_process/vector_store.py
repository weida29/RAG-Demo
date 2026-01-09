"""
向量存储模块

封装 Chroma 向量数据库，提供文档存储和检索功能。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from .loader import Document


class VectorStore:
    """Chroma 向量存储，支持本地持久化"""
    
    DEFAULT_PERSIST_DIR = Path("data/chroma_db")
    DEFAULT_COLLECTION_NAME = "law_articles"
    
    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ):
        """
        初始化向量存储
        
        Args:
            persist_dir: 持久化目录路径
            collection_name: 集合名称
        """
        self.persist_dir = persist_dir or self.DEFAULT_PERSIST_DIR
        self.collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        
        # 确保目录存在
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 Chroma 客户端（持久化模式）
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = None
    
    @property
    def collection(self):
        """获取或创建集合"""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
            )
        return self._collection
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
    ) -> None:
        """
        添加文档到向量存储
        
        Args:
            documents: Document 对象列表
            embeddings: 对应的向量列表
        """
        if not documents:
            return
        
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Chroma 需要 list 类型的 embeddings
        embeddings_list = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
        
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )
    
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        查询最相似的文档
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            where: 元数据过滤条件
            
        Returns:
            结果列表，每个元素包含 id, content, metadata, distance
        """
        # 转换为 list 类型
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        # 整理结果格式
        output = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                })
        
        return output
    
    def count(self) -> int:
        """返回存储的文档数量"""
        return self.collection.count()
    
    def clear(self) -> None:
        """清空集合"""
        self._client.delete_collection(self.collection_name)
        self._collection = None
    
    def exists(self) -> bool:
        """检查集合是否存在且有数据"""
        try:
            return self.count() > 0
        except Exception:
            return False


if __name__ == "__main__":
    # 测试向量存储
    import numpy as np
    
    store = VectorStore(persist_dir=Path("data/chroma_db_test"))
    
    # 创建测试文档
    docs = [
        Document(id="test_1", content="民法典第一条", metadata={"heading": "第一条"}),
        Document(id="test_2", content="民法典第二条", metadata={"heading": "第二条"}),
    ]
    
    # 模拟向量（实际应使用 Embedder）
    embeddings = np.random.rand(2, 384).tolist()
    
    # 添加文档
    store.add_documents(docs, embeddings)
    print(f"文档数量: {store.count()}")
    
    # 查询
    query_emb = np.random.rand(384).tolist()
    results = store.query(query_emb, top_k=2)
    print(f"查询结果: {len(results)} 条")
    for r in results:
        print(f"  - {r['id']}: {r['content']}")
    
    # 清理测试数据
    store.clear()
    print("测试数据已清理")


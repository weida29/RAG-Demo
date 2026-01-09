"""
向量化模块

封装 Sentence Transformers 模型，提供文本向量化功能。
模型权重保存在本地 models/ 目录，避免重复下载。
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np


class Embedder:
    """文本向量化器，使用 Sentence Transformers 模型"""
    
    # 默认使用多语言模型，支持中文
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # 本地模型存储目录（与本文件同级的 models/ 目录）
    MODELS_DIR = Path(__file__).parent / "models"
    
    def __init__(self, model_name: Optional[str] = None, use_local_cache: bool = True):
        """
        初始化向量化器
        
        Args:
            model_name: 模型名称，默认使用多语言 MiniLM 模型
            use_local_cache: 是否使用本地缓存（默认 True）
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.use_local_cache = use_local_cache
        self._model = None  # 延迟加载
    
    @property
    def local_model_path(self) -> Path:
        """获取本地模型存储路径"""
        # 将模型名中的 / 替换为 _，避免路径问题
        safe_name = self.model_name.replace("/", "_")
        return self.MODELS_DIR / safe_name
    
    @property
    def model(self):
        """延迟加载模型，优先从本地加载"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            
            local_path = self.local_model_path
            
            if self.use_local_cache and local_path.exists():
                # 从本地加载模型
                print(f"从本地加载模型: {local_path}")
                self._model = SentenceTransformer(str(local_path))
            else:
                # 从 HuggingFace 下载模型
                print(f"从 HuggingFace 下载模型: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                
                # 保存到本地
                if self.use_local_cache:
                    self._save_model_locally()
        
        return self._model
    
    def _save_model_locally(self) -> None:
        """将模型保存到本地目录"""
        local_path = self.local_model_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"保存模型到本地: {local_path}")
        self._model.save(str(local_path))
        print("模型已保存，下次将从本地加载")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        将文本转换为向量
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            numpy 数组，形状为 (n, dim) 或 (dim,)
        """
        if isinstance(texts, str):
            texts = [texts]
            return self.model.encode(texts, convert_to_numpy=True)[0]
        return self.model.encode(texts, convert_to_numpy=True)
    
    def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        批量向量化文本
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            
        Returns:
            numpy 数组，形状为 (n, dim)
        """
        return self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
    
    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    # 测试向量化器
    embedder = Embedder()
    
    test_texts = [
        "中华人民共和国民法典",
        "自然人的民事权利能力一律平等",
        "法人是具有民事权利能力和民事行为能力的组织",
    ]
    
    print(f"模型: {embedder.model_name}")
    print(f"向量维度: {embedder.dimension}")
    
    embeddings = embedder.embed(test_texts)
    print(f"向量形状: {embeddings.shape}")
    
    # 计算相似度
    from numpy.linalg import norm
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))
    
    print(f"\n文本相似度:")
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  [{i}] vs [{j}]: {sim:.4f}")


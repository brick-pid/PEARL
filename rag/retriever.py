from typing import List, Dict, Any, Tuple, Optional, Union, Literal
from tqdm import tqdm
import json
from collections import Counter
from abc import ABC, abstractmethod
import os
import faiss
import numpy as np
from .embedder import BaseEmbedder
import bm25s


class BaseDataSource(ABC):
    """数据源抽象基类"""
    
    @abstractmethod
    def load_chunks(self) -> List[str]:
        """加载数据"""
        pass
    
    @property
    @abstractmethod
    def result_pool(self) -> List[str]:
        """返回用于最终结果的数据池"""
        pass


class KnowledgeDataSource(BaseDataSource):
    """知识库数据源"""
    
    def __init__(self, knowledge_path: str):
        self.knowledge_path = knowledge_path
        self._knowledges = []
        self._chunks = []
        self.load_chunks()
        
    def load_chunks(self) -> List[str]:
        with open(self.knowledge_path, 'r', encoding='utf-8') as f:
            for line in f:
                knowledge = json.loads(line.strip())
                chunk = knowledge['knowledge_entity'] + '\n' + knowledge['intent']
                knowledge_str = chunk + '\n' + knowledge['content'] + '\n' + knowledge['code_demo']
                self._knowledges.append(knowledge_str)
                self._chunks.append(chunk)
        return self._chunks
    
    @property
    def result_pool(self) -> List[str]:
        return self._knowledges


class CodeDataSource(BaseDataSource):
    """代码数据源"""
    
    def __init__(self, lang: str = 'rkt'):
        self.lang = lang
        self._chunks = []
        self.load_chunks()
        
    def load_chunks(self) -> List[str]:
        from utils.process_utils import get_long_language_name
        from datasets import load_dataset
        
        long_lang = get_long_language_name(self.lang)
        dataset = load_dataset("nuprl/MultiPL-T", split=long_lang)
        self._chunks = dataset['content']
        return self._chunks
    
    @property
    def result_pool(self) -> List[str]:
        return self._chunks


class BaseRetriever(ABC):
    """
    抽象基类定义
    支持两种检索模式：
    1. 文本检索：需要提供 embedder
    2. 向量检索：直接输入向量进行检索
    """
    def __init__(self, 
                 index_path: str,
                 data_source: BaseDataSource,
                 batch_size: int = 64):
        self.index_path = index_path
        self.batch_size = batch_size
        self.data_source = data_source
        self.chunks = self.data_source.load_chunks()
        self.index = self.prepare_index()
    
    @abstractmethod
    def prepare_index(self) -> Any:
        """准备索引"""
        pass
    
    @abstractmethod
    def _retrieve(self, 
                queries: Union[List[str], np.ndarray], 
                top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """检索方法"""
        pass

    def _get_results(self, scores: np.ndarray, indices: np.ndarray) -> List[List[str]]:
        """获取检索结果"""
        assert scores.shape == indices.shape
        assert scores.ndim == 2
        
        pool = self.data_source.result_pool
        results = []
        for score, index in zip(scores, indices):
            r = []
            for s, i in zip(score, index):
                r.append(pool[i])
            results.append(r)
        return results
    
    def retrieve_results(self, queries: Union[List[str], np.ndarray], top_k: int = 5) -> List[List[str]]:
        scores, indices = self._retrieve(queries, top_k)
        return self._get_results(scores, indices)


class DenseRetriever(BaseRetriever):
    """稠密向量检索模型"""
    
    def __init__(self, 
                 index_path: str,
                 data_source: BaseDataSource,
                 embedder: BaseEmbedder,
                 batch_size: int = 64):
        self.embedder = embedder
        super().__init__(index_path, data_source, batch_size)

    def prepare_index(self) -> faiss.Index:
        """Prepare FAISS index: load from cache or build new one."""
        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)
        if self.embedder is None:
            raise ValueError("Embedder is required to build new index")
        return self.build_index()
    
    def build_index(self) -> faiss.Index:
        """Build FAISS index with batch processing."""
        if self.embedder is None:
            raise ValueError("Embedder is required to build index")
            
        index = None
        total_batches = (len(self.chunks) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=total_batches, desc="Building Index") as pbar:
            for i in range(0, len(self.chunks), self.batch_size):
                batch = self.chunks[i:i+self.batch_size]
                embeddings = self.embedder.encode(batch)
                
                if index is None:
                    # 使用 L2 归一化的余弦相似度索引
                    dimension = embeddings.shape[1]
                    index = faiss.IndexFlatIP(dimension)
                    # 确保向量已经归一化
                    faiss.normalize_L2(embeddings)
                else:
                    faiss.normalize_L2(embeddings)
                
                index.add(embeddings)
                pbar.update(1)
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(index, self.index_path)
        return index

    def _retrieve(self, 
                queries: Union[List[str], np.ndarray], 
                top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Base retrieval method supporting both text and vector queries
        
        Args:
            queries: Either a list of text queries or a numpy array of query vectors
            top_k: Number of results to return
            
        Returns:
            Tuple of (scores, indices)
        """
        if isinstance(queries, list):
            if self.embedder is None:
                raise ValueError("Embedder is required for text queries")
            query_vectors = self.embedder.encode(queries)
        else:
            query_vectors = queries
            
        if not isinstance(query_vectors, np.ndarray):
            raise ValueError("Query vectors must be a numpy array")
            
        # 确保查询向量也经过 L2 归一化
        query_vectors = np.float32(query_vectors)
        faiss.normalize_L2(query_vectors)
        scores, indices = self.index.search(query_vectors, top_k)
        
        # 添加检查
        if (indices == -1).any():
            print("Warning: Some queries returned no results")
            
        return scores, indices


class BM25Retriever(BaseRetriever):
    """BM25检索模型"""
    
    def __init__(self, 
                 index_path: str,
                 data_source: BaseDataSource,
                 tokenizer_path: Optional[str] = None,
                 method: str = "lucene",
                 k1: float = 1.5,
                 b: float = 0.75,
                 batch_size: int = 64):
        """
        初始化BM25检索器
        
        Args:
            index_path: BM25索引保存路径
            data_source: 数据源实例
            tokenizer_path: 分词器保存路径(可选)
            method: BM25变体 ("robertson", "atire", "bm25l", "bm25+", "lucene")
            k1: BM25参数k1
            b: BM25参数b
            batch_size: 批处理大小
        """
        import bm25s
        self.tokenizer_path = tokenizer_path or os.path.join(os.path.dirname(index_path), "tokenizer")
        self.tokenizer = bm25s.tokenization.Tokenizer()
        self.bm25_index = bm25s.BM25(method=method, k1=k1, b=b)
        super().__init__(index_path, data_source, batch_size)

    def prepare_index(self) -> Any:
        """准备BM25索引"""
        if os.path.exists(self.index_path):
            # 如果索引存在，从保存的文件加载
            self.bm25_index = bm25s.BM25.load(self.index_path)
            return self.bm25_index
            
        # 对文档进行分词
        corpus_tokens = self.tokenizer.tokenize(self.chunks)
        # 构建索引
        self.bm25_index.index(corpus_tokens)
        # 保存索引和分词器
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self.bm25_index.save(self.index_path, corpus=self.chunks)
        self.tokenizer.save_vocab(self.tokenizer_path)
        return self.bm25_index

    def _retrieve(self, 
                queries: Union[List[str], np.ndarray], 
                top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行检索
        
        Args:
            queries: 查询文本列表或查询向量
            top_k: 返回的结果数量
            
        Returns:
            (scores, indices): 分数和对应的文档索引
        """
        if isinstance(queries, np.ndarray):
            raise ValueError("BM25检索器不支持向量查询")
            
        # 对查询进行分词
        query_tokens = self.tokenizer.tokenize(queries)
        
        # 执行检索
        indices, scores = self.bm25_index.retrieve(
            query_tokens,
            k=top_k,
            return_as="tuple"
        )

        return scores, indices

# self.bm25_index.retrieve(query_tokens,k=top_k,return_as="tuple")

class RetrieverFactory:
    """检索器工厂"""
    
    @staticmethod
    def create(retriever_type: Literal["dense", "bm25"],
              data_source_type: Literal["knowledge", "code"],
              index_path: str,
              data_source: str,
              embedder: BaseEmbedder = None) -> BaseRetriever:
        """
        创建检索器实例
        
        Args:
            retriever_type: 检索模型类型
            data_source_type: 数据源类型
            index_path: 索引路径
            data_source_args: 数据源参数
            retriever_args: 检索器参数
            
        Returns:
            BaseRetriever: 检索器实例
        """
        # 创建数据源
        if data_source_type == "knowledge":
            data_source = KnowledgeDataSource(data_source)
        elif data_source_type == "code":
            data_source = CodeDataSource(data_source)
        else:
            raise ValueError(f"Unknown data source type: {data_source_type}")
            
        # 创建检索器
        if retriever_type == "dense":
            return DenseRetriever(index_path, data_source, embedder)
        elif retriever_type == "bm25":
            return BM25Retriever(index_path, data_source)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")


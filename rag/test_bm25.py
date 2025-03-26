import pytest
import os
import tempfile
import json
import numpy as np
from rag.retriever import (
    KnowledgeDataSource, 
    CodeDataSource,
    DenseRetriever,
    BM25Retriever,
    RetrieverFactory,
    BaseEmbedder
)

# 模拟的Embedder用于测试
class MockEmbedder():
    def encode(self, texts):
        # 为每个文本生成一个固定维度的随机向量
        return np.random.randn(len(texts), 384).astype(np.float32)


@pytest.fixture
def index_dir():
    # 创建临时索引目录
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

# 测试KnowledgeDataSource
def test_knowledge_data_source():
    knowledge_path = "/home/sjw/ljb/lr_rag/documents/knowledge/lua/lua_knowledge_v3_mini.jsonl"
    source = KnowledgeDataSource(knowledge_path)
    chunks = source.load_chunks()
    pool = source.result_pool
    
    assert len(chunks) > 0
    assert len(pool) > 0
    assert len(chunks) == len(pool)  # 对于代码数据源，chunks和pool应该相同

# 测试CodeDataSource
def test_code_data_source():
    source = CodeDataSource(lang='rkt')
    chunks = source.load_chunks()
    pool = source.result_pool
    
    assert len(chunks) > 0
    assert len(pool) > 0
    assert len(chunks) == len(pool)  # 对于代码数据源，chunks和pool应该相同

# 测试DenseRetriever
def test_dense_retriever(index_dir):
    knowledge_path = "/home/sjw/ljb/lr_rag/documents/knowledge/lua/lua_knowledge_v3_mini.jsonl"
    source = KnowledgeDataSource(knowledge_path)
    embedder = MockEmbedder()
    index_path = os.path.join(index_dir, "dense.index")
    
    # 创建检索器
    retriever = DenseRetriever(
        index_path=index_path,
        data_source=source,
        embedder=embedder
    )
    
    # 测试检索
    queries = ["测试查询1", "测试查询2"]
    results = retriever.retrieve_results(queries, top_k=2)
    
    assert len(results) == 2  # 两个查询
    assert len(results[0]) == 2  # 每个查询返回2个结果
    assert all(isinstance(r, str) for r in results[0])

# 测试BM25Retriever
def test_bm25_retriever(index_dir):
    knowledge_path = "/home/sjw/ljb/lr_rag/documents/knowledge/lua/lua_knowledge_v3_mini.jsonl"
    source = KnowledgeDataSource(knowledge_path)
    index_path = os.path.join(index_dir, "bm25.index")
    
    # 创建检索器
    retriever = BM25Retriever(
        index_path=index_path,
        data_source=source,
        tokenizer_path=os.path.join(index_dir, "tokenizer")
    )
    
    # 测试检索
    breakpoint()
    queries = ["lua is a programming language", "lua for loop syntax"]
    results = retriever.retrieve_results(queries, top_k=2)
    
    breakpoint()
    assert len(results) == 2
    assert len(results[0]) == 2
    assert all(isinstance(r, str) for r in results[0])

# 测试RetrieverFactory
def test_retriever_factory(index_dir):
    knowledge_path = "/home/sjw/ljb/lr_rag/documents/knowledge/lua/lua_knowledge_v3_mini.jsonl"
    # 测试创建Dense检索器
    dense_retriever = RetrieverFactory.create(
        retriever_type="dense",
        data_source_type="knowledge",
        index_path=os.path.join(index_dir, "dense.index"),
        data_source_args={"knowledge_path": knowledge_path},
        retriever_args={"embedder": MockEmbedder()}
    )
    assert isinstance(dense_retriever, DenseRetriever)
    
    # 测试创建BM25检索器
    bm25_retriever = RetrieverFactory.create(
        retriever_type="bm25",
        data_source_type="knowledge",
        index_path=os.path.join(index_dir, "bm25.index"),
        data_source_args={"knowledge_path": knowledge_path}
    )
    assert isinstance(bm25_retriever, BM25Retriever)

# 测试错误处理
def test_error_handling(index_dir):
    knowledge_path = "/home/sjw/ljb/lr_rag/documents/knowledge/lua/lua_knowledge_v3_mini.jsonl"
    # 测试无效的检索器类型
    with pytest.raises(ValueError):
        RetrieverFactory.create(
            retriever_type="invalid",
            data_source_type="knowledge",
            index_path="test.index",
            data_source_args={"knowledge_path": knowledge_path}
        )
    
    # 测试无效的数据源类型
    with pytest.raises(ValueError):
        RetrieverFactory.create(
            retriever_type="dense",
            data_source_type="invalid",
            index_path="test.index"
        )
    
    # 测试BM25检索器不支持向量查询
    source = KnowledgeDataSource(knowledge_path)
    retriever = BM25Retriever(
        index_path=os.path.join(index_dir, "bm25.index"),
        data_source=source
    )
    with pytest.raises(ValueError):
        retriever._retrieve(np.random.randn(2, 384))
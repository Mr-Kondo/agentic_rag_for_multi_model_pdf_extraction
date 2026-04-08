"""Tests for src/core/embedder.py — embedder backends and factory."""

from unittest.mock import MagicMock, patch

import pytest


# ── SentenceTransformerEmbedder ──────────────────────────────────────────────


def test_st_embedder_encode_passages():
    """SentenceTransformerEmbedder prepends passage prefix before encoding."""
    from src.core.embedder import SentenceTransformerEmbedder

    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1] * 384, [0.2] * 384])

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
        embedder.encode_passages(["hello", "world"])

    call_args = mock_model.encode.call_args
    texts_sent = call_args[0][0]
    assert texts_sent[0].startswith("passage: ")
    assert texts_sent[1].startswith("passage: ")


def test_st_embedder_encode_query():
    """SentenceTransformerEmbedder prepends query prefix before encoding."""
    from src.core.embedder import SentenceTransformerEmbedder

    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1] * 384])

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
        embedder.encode_query("test query")

    call_args = mock_model.encode.call_args
    texts_sent = call_args[0][0]
    assert texts_sent[0] == "query: test query"


def test_st_embedder_custom_prefixes():
    """SentenceTransformerEmbedder respects custom prefix configuration."""
    from src.core.embedder import SentenceTransformerEmbedder

    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.0] * 384])

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        embedder = SentenceTransformerEmbedder(
            "some-model",
            query_prefix="クエリ: ",
            passage_prefix="文章: ",
        )
        embedder.encode_query("日本語クエリ")

    texts_sent = mock_model.encode.call_args[0][0]
    assert texts_sent[0] == "クエリ: 日本語クエリ"


def test_st_embedder_embedding_dim():
    """SentenceTransformerEmbedder returns correct embedding dimension."""
    from src.core.embedder import SentenceTransformerEmbedder

    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 1024

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        embedder = SentenceTransformerEmbedder("some-large-model")

    assert embedder.embedding_dim == 1024


# ── OllamaEmbedder ───────────────────────────────────────────────────────────


def _make_ollama_client_mock(dim: int = 1024):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.embeddings = [[0.1] * dim]
    mock_client.embed.return_value = mock_response
    return mock_client


def test_ollama_embedder_encode_passages():
    """OllamaEmbedder prepends Japanese passage prefix."""
    import ollama

    from src.core.embedder import OllamaEmbedder

    mock_client = _make_ollama_client_mock()
    mock_client.embed.return_value.embeddings = [[0.1] * 1024, [0.2] * 1024]

    with patch.object(ollama, "Client", return_value=mock_client):
        embedder = OllamaEmbedder("kun432/cl-nagoya-ruri-large")
        embedder.encode_passages(["テスト1", "テスト2"])

    call_kwargs = mock_client.embed.call_args
    inputs = call_kwargs[1]["input"] if call_kwargs[1] else call_kwargs[0][1]
    assert inputs[0].startswith("文章: ")
    assert inputs[1].startswith("文章: ")


def test_ollama_embedder_encode_query():
    """OllamaEmbedder prepends Japanese query prefix."""
    import ollama

    from src.core.embedder import OllamaEmbedder

    mock_client = _make_ollama_client_mock()

    with patch.object(ollama, "Client", return_value=mock_client):
        embedder = OllamaEmbedder("kun432/cl-nagoya-ruri-large")
        embedder.encode_query("STIVのメリットは？")

    call_kwargs = mock_client.embed.call_args
    inputs = call_kwargs[1]["input"] if call_kwargs[1] else call_kwargs[0][1]
    assert inputs[0] == "クエリ: STIVのメリットは？"


def test_ollama_embedder_dimension_probe():
    """OllamaEmbedder probes dimension lazily on first access."""
    import ollama

    from src.core.embedder import OllamaEmbedder

    probe_response = MagicMock()
    probe_response.embeddings = [[0.0] * 1024]
    mock_client = MagicMock()
    mock_client.embed.return_value = probe_response

    with patch.object(ollama, "Client", return_value=mock_client):
        embedder = OllamaEmbedder("kun432/cl-nagoya-ruri-large")
        dim = embedder.embedding_dim

    assert dim == 1024


def test_ollama_embedder_dimension_probe_empty_raises():
    """OllamaEmbedder raises ValueError when Ollama returns empty embeddings."""
    import ollama

    from src.core.embedder import OllamaEmbedder

    empty_response = MagicMock()
    empty_response.embeddings = []
    mock_client = MagicMock()
    mock_client.embed.return_value = empty_response

    with patch.object(ollama, "Client", return_value=mock_client):
        embedder = OllamaEmbedder("not-loaded-model")
        with pytest.raises(ValueError, match="empty embeddings"):
            _ = embedder.embedding_dim


def test_ollama_embedder_batching():
    """OllamaEmbedder splits large input lists into batches."""
    import ollama

    from src.core.embedder import OllamaEmbedder

    batch_response = MagicMock()
    batch_response.embeddings = [[0.1] * 1024] * 2  # 2 per batch
    mock_client = MagicMock()
    mock_client.embed.return_value = batch_response

    with patch.object(ollama, "Client", return_value=mock_client):
        embedder = OllamaEmbedder("model", batch_size=2)
        results = embedder.encode_passages(["a", "b", "c", "d"])

    # 4 texts with batch_size=2 → 2 embed() calls
    assert mock_client.embed.call_count == 2
    assert len(results) == 4


# ── create_embedder factory ──────────────────────────────────────────────────


def test_create_embedder_st_backend():
    """Factory returns SentenceTransformerEmbedder for sentence_transformers backend."""
    from src.core.embedder import SentenceTransformerEmbedder, create_embedder

    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        emb = create_embedder("intfloat/multilingual-e5-small", backend="sentence_transformers")

    assert isinstance(emb, SentenceTransformerEmbedder)


def test_create_embedder_ollama_backend():
    """Factory returns OllamaEmbedder for ollama backend."""
    import ollama

    from src.core.embedder import OllamaEmbedder, create_embedder

    mock_client = _make_ollama_client_mock()

    with patch.object(ollama, "Client", return_value=mock_client):
        emb = create_embedder("kun432/cl-nagoya-ruri-large", backend="ollama")

    assert isinstance(emb, OllamaEmbedder)


def test_create_embedder_unknown_backend():
    """Factory raises ValueError for unknown backend."""
    from src.core.embedder import create_embedder

    with pytest.raises(ValueError, match="Unknown embedder backend"):
        create_embedder("some-model", backend="foobar")


def test_create_embedder_custom_prefixes_override():
    """Factory passes custom prefixes to the embedder."""
    from src.core.embedder import SentenceTransformerEmbedder, create_embedder

    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.0] * 384])

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        emb = create_embedder(
            "some-model",
            backend="sentence_transformers",
            query_prefix="Q: ",
            passage_prefix="P: ",
        )
        emb.encode_query("test")

    texts_sent = mock_model.encode.call_args[0][0]
    assert texts_sent[0] == "Q: test"

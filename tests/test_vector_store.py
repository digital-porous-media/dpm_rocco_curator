"""Tests for vector store creation and embedding batch handling.

Tests that the vector store can handle partial embedding failures gracefully
and maintains correct document/embedding alignment. This catches regressions
from API changes (e.g., switching to SambaNova embeddings).

Run with: pytest tests/test_vector_store.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.retriever.retriever import VectorStoreManager
from src.ingestor.embedder import DocumentEmbedder


@pytest.fixture
def mock_embedder():
    """Provide a mocked DocumentEmbedder."""
    embedder = MagicMock(spec=DocumentEmbedder)
    embedder.embeddings = MagicMock()
    return embedder


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(page_content=f"Document {i}", metadata={"index": i})
        for i in range(20)
    ]


class TestVectorStoreCreation:
    """Test basic vector store creation."""

    def test_create_from_documents_success(self, mock_embedder, sample_documents):
        """Vector store should create successfully with all documents embedded."""
        # Mock successful embedding
        mock_embedder.embed_documents = MagicMock(
            return_value=[[0.1 * i, 0.2 * i] for i in range(20)]
        )

        with patch("src.retriever.retriever.FAISS.from_embeddings") as mock_faiss:
            mock_faiss.return_value = MagicMock(spec=FAISS)

            manager = VectorStoreManager(embedder=mock_embedder)
            vs = manager.create_from_documents(sample_documents)

            assert vs is not None
            # Verify from_embeddings was called with 20 text-embedding pairs
            mock_faiss.assert_called_once()
            call_args = mock_faiss.call_args
            text_embeddings = call_args.kwargs["text_embeddings"]
            assert len(text_embeddings) == 20


class TestVectorStoreBatching:
    """Test that embedding is batched correctly."""

    def test_batches_documents_into_groups_of_32(self, mock_embedder, sample_documents):
        """Documents should be embedded in batches of 32."""
        # Create 100 documents to trigger multiple batches
        docs = [
            Document(page_content=f"Doc {i}", metadata={"index": i})
            for i in range(100)
        ]

        embedding_calls = []

        def track_embeddings(texts):
            """Track embedding calls to verify batching."""
            embedding_calls.append(len(texts))
            return [[0.1, 0.2] for _ in texts]

        mock_embedder.embed_documents = track_embeddings

        with patch("src.retriever.retriever.FAISS.from_embeddings") as mock_faiss:
            mock_faiss.return_value = MagicMock(spec=FAISS)

            manager = VectorStoreManager(embedder=mock_embedder)
            manager.create_from_documents(docs)

            # Should have batched into: 32 + 32 + 32 + 4
            assert embedding_calls == [32, 32, 32, 4]

    def test_batch_smaller_than_32(self, mock_embedder):
        """Batching should handle documents < 32 correctly."""
        docs = [
            Document(page_content=f"Doc {i}", metadata={"index": i})
            for i in range(10)
        ]

        embedding_calls = []

        def track_embeddings(texts):
            embedding_calls.append(len(texts))
            return [[0.1, 0.2] for _ in texts]

        mock_embedder.embed_documents = track_embeddings

        with patch("src.retriever.retriever.FAISS.from_embeddings") as mock_faiss:
            mock_faiss.return_value = MagicMock(spec=FAISS)

            manager = VectorStoreManager(embedder=mock_embedder)
            manager.create_from_documents(docs)

            # Should process all 10 in one batch
            assert embedding_calls == [10]


class TestVectorStoreFailureHandling:
    """Test graceful handling of embedding failures."""

    def test_skip_batch_with_mismatched_embedding_count(self, mock_embedder, sample_documents):
        """If a batch embedding fails (returns fewer embeddings), skip the batch."""
        # Mock: batch 1 succeeds, batch 2 fails (returns empty), batch 3 succeeds
        batch_results = [
            [[0.1, 0.2] for _ in range(20)],  # Batch 1: success
            [],  # Batch 2: failure
        ]
        mock_embedder.embed_documents = MagicMock(side_effect=batch_results)

        with patch("src.retriever.retriever.FAISS.from_embeddings") as mock_faiss:
            mock_faiss.return_value = MagicMock(spec=FAISS)

            manager = VectorStoreManager(embedder=mock_embedder)
            vs = manager.create_from_documents(sample_documents)

            # Should succeed but with fewer documents
            assert vs is not None
            call_args = mock_faiss.call_args
            text_embeddings = call_args.kwargs["text_embeddings"]
            # Only batch 1 should have been included
            assert len(text_embeddings) == 20

    def test_skip_batch_on_exception(self, mock_embedder, sample_documents):
        """If embedding raises an exception, skip that batch and continue."""
        # Mock: batch 1 succeeds, batch 2 raises exception, batch 3 succeeds
        def mock_embed_with_error(texts):
            if mock_embed_with_error.call_count == 2:
                mock_embed_with_error.call_count += 1
                raise RuntimeError("API timeout")
            mock_embed_with_error.call_count += 1
            return [[0.1, 0.2] for _ in texts]

        mock_embed_with_error.call_count = 1
        mock_embedder.embed_documents = mock_embed_with_error

        with patch("src.retriever.retriever.FAISS.from_embeddings") as mock_faiss:
            mock_faiss.return_value = MagicMock(spec=FAISS)

            manager = VectorStoreManager(embedder=mock_embedder)
            # Should not raise, should gracefully skip failed batch
            vs = manager.create_from_documents(sample_documents)

            assert vs is not None

    def test_raise_error_if_no_documents_embedded(self, mock_embedder, sample_documents):
        """If no documents can be embedded, raise a clear error."""
        # Mock: all batches fail
        mock_embedder.embed_documents = MagicMock(return_value=[])

        with patch("src.retriever.retriever.FAISS.from_embeddings"):
            manager = VectorStoreManager(embedder=mock_embedder)

            with pytest.raises(ValueError, match="No documents were successfully embedded"):
                manager.create_from_documents(sample_documents)


class TestVectorStoreDocumentAlignment:
    """Test that documents and embeddings always stay aligned."""

    def test_metadata_preserved_after_embedding(self, mock_embedder):
        """Document metadata should be preserved even if some documents fail to embed."""
        docs = [
            Document(page_content=f"Doc {i}", metadata={"id": i, "title": f"Title {i}"})
            for i in range(10)
        ]

        mock_embedder.embed_documents = MagicMock(
            return_value=[[0.1, 0.2] for _ in range(10)]
        )

        with patch("src.retriever.retriever.FAISS.from_embeddings") as mock_faiss:
            mock_faiss.return_value = MagicMock(spec=FAISS)

            manager = VectorStoreManager(embedder=mock_embedder)
            manager.create_from_documents(docs)

            # Verify metadata was passed correctly
            call_args = mock_faiss.call_args
            metadatas = call_args.kwargs["metadatas"]
            assert len(metadatas) == 10
            assert metadatas[0] == {"id": 0, "title": "Title 0"}
            assert metadatas[9] == {"id": 9, "title": "Title 9"}

    def test_text_embedding_pairs_aligned(self, mock_embedder):
        """Text and embedding vectors should be correctly paired."""
        docs = [
            Document(page_content=f"Content {i}", metadata={"index": i})
            for i in range(5)
        ]

        embeddings = [[float(i), float(i + 1)] for i in range(5)]
        mock_embedder.embed_documents = MagicMock(return_value=embeddings)

        with patch("src.retriever.retriever.FAISS.from_embeddings") as mock_faiss:
            mock_faiss.return_value = MagicMock(spec=FAISS)

            manager = VectorStoreManager(embedder=mock_embedder)
            manager.create_from_documents(docs)

            # Verify text-embedding pairing
            call_args = mock_faiss.call_args
            text_embeddings = call_args.kwargs["text_embeddings"]

            for i, (text, embedding) in enumerate(text_embeddings):
                assert text == f"Content {i}"
                assert embedding == [float(i), float(i + 1)]


class TestVectorStorePartialFailure:
    """Test the realistic case of partial batch failures (like with SambaNova API)."""

    def test_179_documents_102_embeddings_scenario(self, mock_embedder):
        """
        Real-world scenario: 179 documents uploaded, but embedding API fails on 77.
        This was the actual bug that prompted these tests.
        """
        # Create 179 documents (simulating a multi-page PDF)
        docs = [
            Document(page_content=f"Page {i} content", metadata={"chunk_index": i})
            for i in range(179)
        ]

        # Mock: batches succeed/fail to simulate API issues
        batch_results = [
            [[0.1, 0.2] for _ in range(32)],  # Batch 0: success (0-31)
            [[0.1, 0.2] for _ in range(32)],  # Batch 1: success (32-63)
            [[0.1, 0.2] for _ in range(32)],  # Batch 2: success (64-95)
            [],  # Batch 3: failure (96-127)
            [[0.1, 0.2] for _ in range(6)],  # Batch 4: partial (128-133)
            [],  # Batch 5: failure (134-165)
            [],  # Batch 6: failure (166-177)
            [[0.1, 0.2] for _ in range(1)],  # Batch 7: success (178)
        ]

        mock_embedder.embed_documents = MagicMock(side_effect=batch_results)

        with patch("src.retriever.retriever.FAISS.from_embeddings") as mock_faiss:
            mock_faiss.return_value = MagicMock(spec=FAISS)

            manager = VectorStoreManager(embedder=mock_embedder)
            # Should not raise ValueError about mismatched counts
            vs = manager.create_from_documents(docs)

            assert vs is not None

            # Verify only successfully embedded docs were included
            call_args = mock_faiss.call_args
            text_embeddings = call_args.kwargs["text_embeddings"]
            # 32 + 32 + 32 + 6 + 1 = 103 (close to the 102 from the real bug)
            assert len(text_embeddings) == 103

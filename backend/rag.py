"""
rag.py — FAISS-backed RAG module with sentence-transformer embeddings.

Provides DocumentLoader (ingest) and RAGRetriever (query-time) classes.
Telecom seed documents are embedded on first run; subsequent runs reload
the persisted FAISS index for sub-millisecond retrieval.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — these are heavy; only loaded when the module is first used.
# ---------------------------------------------------------------------------

def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer  # type: ignore
    return SentenceTransformer(config.rag.embedding_model)


def _load_faiss():
    import faiss  # type: ignore
    return faiss


# ---------------------------------------------------------------------------
# Built-in telecom seed documents
# These are embedded into the FAISS index if no external docs are found.
# ---------------------------------------------------------------------------

TELECOM_SEED_DOCS: List[str] = [
    # Bundles
    "The 400MB hourly data bundle costs $1 and is valid for 1 hour. "
    "It is the cheapest available bundle for light browsing.",

    "The 1GB daily bundle costs $3 and is valid for 24 hours. "
    "Good for moderate daily internet usage including social media.",

    "The 5GB weekly bundle costs $10 and is valid for 7 days. "
    "Recommended for heavy users who stream videos or work remotely.",

    "The 20GB monthly bundle costs $30 and is valid for 30 days. "
    "Best value for month-long high-volume data users.",

    "Unlimited daily bundle costs $5 and offers unlimited data for 24 hours "
    "with fair-use policy of 10GB at full speed, then throttled to 1Mbps.",

    # Voice and SMS
    "The standard local call rate is $0.10 per minute. "
    "International calls vary by destination, starting from $0.20 per minute.",

    "SMS bundles: 100 SMS for $1 valid 7 days, 500 SMS for $3 valid 30 days.",

    "Roaming packages are available for 50 countries. "
    "Activate before travel via the app or by calling customer service.",

    # Account management
    "To check your balance, dial *123# or log in to the MyTelecom app. "
    "Balance is updated in real time.",

    "Top-up options include credit card, debit card, bank transfer, and voucher codes. "
    "Minimum top-up amount is $1.",

    "Auto-renew can be enabled for any recurring bundle. "
    "You will be notified 24 hours before renewal if the balance is insufficient.",

    "Bundle activation is immediate after purchase. "
    "You will receive an SMS confirmation within 30 seconds.",

    # Support
    "Customer service is available 24/7 via phone (dial 100), live chat, "
    "and email at support@telecom.example.",

    "Network outages are posted on the status page at status.telecom.example "
    "and on the official Twitter account @TelecomStatus.",

    "SIM replacement for lost or damaged SIM cards requires visiting a store "
    "with a valid government-issued ID. The fee is $2.",

    "Port-in offers allow new customers to keep their existing number. "
    "The porting process takes up to 24 hours after submission.",

    # Device & APN
    "APN settings for Android: Name=TelecomNet, APN=internet.telecom, "
    "Username and Password left blank.",

    "APN settings for iPhone: Go to Settings > Cellular > Cellular Data Network. "
    "Set APN to internet.telecom.",

    "Wi-Fi calling is supported on compatible devices. Enable in phone settings "
    "under Wi-Fi Calling. Charges apply at standard rates.",

    # Billing
    "Paper bills are sent monthly. Switch to e-billing in account settings "
    "to receive PDF invoices by email.",

    "Disputed charges can be reported within 60 days of the bill date. "
    "Call 100 or submit a ticket at support.telecom.example.",

    "Late payment fee is $2 applied after 15 days past the due date. "
    "Service may be suspended after 30 days of non-payment.",

    # Qobox company information
    "Company Name: Quality Outside The Box (Qobox). "
    "Qobox is an Indian software quality assurance and testing services company "
    "that focuses on delivering reliable, scalable, and high-performance software systems. "
    "The company specializes in software testing, automation frameworks, performance engineering, "
    "and quality consulting services for enterprise applications.",

    "Qobox was established in 2020 and operates as a private limited company registered in India. "
    "The company is headquartered in Chennai, Tamil Nadu, India. "
    "The company has two directors: Gangapuram Jayanthi and Amalodbhavi Mrudula Ranjan Damavarapu.",

    "Qobox operates in the Information Technology services industry, "
    "particularly in the software testing and quality assurance domain. "
    "Qobox provides services to multiple industries including Healthcare, Insurance, Retail, "
    "Financial services, Telecommunications, and Pharmaceutical companies.",

    "Qobox core services include: Software Testing — manual testing for web, mobile, and enterprise applications. "
    "Automation Testing — developing automated test frameworks to accelerate testing cycles. "
    "Performance Testing — testing software under heavy load conditions to ensure system stability. "
    "Security Testing — identifying vulnerabilities and protecting applications from security threats. "
    "API Testing — validating backend APIs for reliability and performance. "
    "QA Consulting — helping organizations implement effective testing strategies and quality frameworks.",

    "Qobox works with modern testing technologies and tools such as test automation frameworks, "
    "Continuous Integration and Continuous Deployment CI/CD pipelines, performance testing tools, "
    "security testing tools, and enterprise testing platforms.",

    "The mission of Qobox is to improve the reliability and quality of software systems by implementing "
    "modern testing methodologies, automation solutions, and performance engineering practices. "
    "The company aims to help organizations deliver high-quality software faster while reducing production issues "
    "and improving user experience.",

    "Qobox follows a quality-driven approach that integrates testing throughout the software development lifecycle. "
    "Their approach includes early defect detection, automated testing strategies, and continuous monitoring "
    "of application performance. "
    "Qobox promotes a culture focused on technical excellence, innovation, and continuous improvement. "
    "The company encourages its teams to adopt new testing technologies and methodologies.",

    "Since its founding, Qobox has expanded its service offerings and continues to support organizations "
    "in improving their software development and release processes through professional quality assurance services. "
    "The company employs professionals specializing in software testing, automation engineering, and quality consulting. "
    "The company aims to expand its capabilities in test automation, AI-assisted testing, "
    "and DevOps-driven quality engineering solutions.",
]


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------

class DocumentLoader:
    """
    Loads, chunks, and indexes text documents for RAG retrieval.

    Usage:
        loader = DocumentLoader()
        loader.load_from_directory("./docs")   # optional external docs
        loader.build_index()
    """

    def __init__(self) -> None:
        self._raw_chunks: List[str] = []

    def load_seed_documents(self) -> None:
        """Load the built-in telecom knowledge base."""
        self._raw_chunks.extend(TELECOM_SEED_DOCS)
        logger.info("Loaded %d seed documents", len(TELECOM_SEED_DOCS))

    def load_from_directory(self, directory: str) -> None:
        """
        Load all .txt files from the given directory, chunk them, and add
        to the internal document pool.

        Args:
            directory: Path to directory containing .txt knowledge-base files.
        """
        doc_dir = Path(directory)
        if not doc_dir.exists():
            logger.warning("Docs directory %s does not exist — skipping", directory)
            return

        txt_files = list(doc_dir.glob("*.txt"))
        if not txt_files:
            logger.warning("No .txt files found in %s", directory)
            return

        for fpath in txt_files:
            try:
                text = fpath.read_text(encoding="utf-8")
                chunks = self.chunk_text(text)
                self._raw_chunks.extend(chunks)
                logger.info("Loaded %d chunks from %s", len(chunks), fpath.name)
            except Exception as exc:
                logger.error("Failed to load %s: %s", fpath, exc)

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping fixed-size word chunks.

        Args:
            text: Raw document text.

        Returns:
            List of string chunks.
        """
        words = text.split()
        chunk_size = config.rag.chunk_size
        overlap = config.rag.chunk_overlap
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def get_chunks(self) -> List[str]:
        """Return the accumulated raw text chunks."""
        return list(self._raw_chunks)

    def build_index(self) -> "FAISSIndex":
        """
        Embed all loaded chunks and build a FAISS index.

        Returns:
            FAISSIndex instance ready for retrieval.
        """
        if not self._raw_chunks:
            raise ValueError("No documents loaded. Call load_seed_documents() first.")
        return FAISSIndex.build(self._raw_chunks)


# ---------------------------------------------------------------------------
# FAISSIndex
# ---------------------------------------------------------------------------

class FAISSIndex:
    """
    Wraps a FAISS flat L2 index together with the text chunks it indexes.

    Supports persistence to disk so the index is not rebuilt on every start.
    """

    def __init__(
        self,
        index,          # faiss.Index
        chunks: List[str],
        model,          # SentenceTransformer
    ) -> None:
        self._index = index
        self._chunks = chunks
        self._model = model

    @classmethod
    def build(cls, chunks: List[str]) -> "FAISSIndex":
        """
        Embed chunks and create a new FAISS IndexFlatIP (inner-product / cosine).

        Args:
            chunks: List of text strings to embed and index.

        Returns:
            Populated FAISSIndex.
        """
        faiss = _load_faiss()
        model = _load_sentence_transformer()

        logger.info("Embedding %d chunks (this may take a moment)…", len(chunks))
        embeddings = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)   # cosine similarity via normalized vectors
        index.add(embeddings)
        logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)
        return cls(index, chunks, model)

    @classmethod
    def load(cls, path: str) -> "FAISSIndex":
        """
        Load a persisted index from disk.

        Args:
            path: Directory where index.faiss and chunks.pkl were saved.

        Returns:
            Loaded FAISSIndex.

        Raises:
            FileNotFoundError: If the index files do not exist.
        """
        faiss = _load_faiss()
        model = _load_sentence_transformer()

        index_file = os.path.join(path, "index.faiss")
        chunks_file = os.path.join(path, "chunks.pkl")

        if not os.path.exists(index_file) or not os.path.exists(chunks_file):
            raise FileNotFoundError(f"No saved index found at {path}")

        index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)

        logger.info("FAISS index loaded from %s (%d vectors)", path, index.ntotal)
        return cls(index, chunks, model)

    def save(self, path: str) -> None:
        """
        Persist the index to disk.

        Args:
            path: Directory to write index.faiss and chunks.pkl.
        """
        faiss = _load_faiss()
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info("FAISS index saved to %s", path)

    def search(
        self, query: str, top_k: int = None, threshold: float = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return. Defaults to config.rag.top_k.
            threshold: Minimum cosine similarity. Defaults to config.rag.similarity_threshold.

        Returns:
            List of (chunk_text, score) tuples sorted by descending score.
        """
        top_k = top_k or config.rag.top_k
        threshold = threshold if threshold is not None else config.rag.similarity_threshold

        q_emb = self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )
        q_emb = np.array(q_emb, dtype=np.float32)

        scores, indices = self._index.search(q_emb, top_k)

        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < threshold:
                continue
            results.append((self._chunks[idx], float(score)))

        logger.debug(
            "RAG search returned %d results for query: %s",
            len(results),
            query[:60],
        )
        return results


# ---------------------------------------------------------------------------
# RAGRetriever — high-level interface used by the LLM client
# ---------------------------------------------------------------------------

class RAGRetriever:
    """
    High-level retriever that manages FAISSIndex lifecycle.

    On construction it will:
    1. Try to load a persisted index from disk.
    2. If none exists, build one from seed + external docs and persist it.

    Usage:
        retriever = RAGRetriever()
        await retriever.initialize()
        context_chunks = retriever.retrieve("cheap internet bundle")
    """

    def __init__(self) -> None:
        self._index: Optional[FAISSIndex] = None

    def initialize(self) -> None:
        """
        Build or reload the FAISS index.  Call once at startup (synchronous,
        acceptable at boot time before the server accepts connections).
        """
        index_path = config.rag.index_path
        try:
            self._index = FAISSIndex.load(index_path)
            logger.info("RAG index loaded from cache at %s", index_path)
        except FileNotFoundError:
            logger.info("No cached RAG index found — building from documents…")
            loader = DocumentLoader()
            loader.load_seed_documents()
            loader.load_from_directory(config.rag.docs_directory)
            self._index = loader.build_index()
            self._index.save(index_path)

    def retrieve(self, query: str) -> List[str]:
        """
        Return the top-k most relevant text chunks for the given query.

        Args:
            query: The user's current utterance or LLM prompt fragment.

        Returns:
            List of relevant text strings (may be empty if index not built
            or no results exceed the similarity threshold).
        """
        if self._index is None:
            logger.warning("RAGRetriever.retrieve called before initialize()")
            return []

        results = self._index.search(query)
        return [chunk for chunk, _score in results]

    def format_context(self, query: str) -> str:
        """
        Build the RAG context block to inject into the LLM prompt.

        Args:
            query: The user's utterance.

        Returns:
            Formatted context string, or empty string if no relevant docs.
        """
        chunks = self.retrieve(query)
        if not chunks:
            return ""
        joined = "\n---\n".join(chunks)
        return f"Relevant knowledge:\n{joined}"

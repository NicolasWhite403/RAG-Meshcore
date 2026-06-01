import os
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


DATA_DIR = Path(__file__).parent / "data"
STORAGE_DIR = Path(__file__).parent / "storage"
FORCE_REBUILD = False


# Embeddings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Optional chunking control (important for multi-doc behavior)
Settings.text_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
)

# LLM
Settings.llm = Ollama(
    model="tinyllama",
    request_timeout=120.0,
    context_window=2048,
    additional_kwargs={
        "num_gpu": 0,
        "num_predict": 200,
    },
)


def build_index():
    if FORCE_REBUILD and STORAGE_DIR.exists():
        import shutil
        shutil.rmtree(STORAGE_DIR)

    if STORAGE_DIR.exists():
        print("\nLoading existing index")
        storage_context = StorageContext.from_defaults(
            persist_dir=str(STORAGE_DIR)
        )
        return load_index_from_storage(storage_context)

    print("\nBuilding index from data folder")

    docs = SimpleDirectoryReader(str(DATA_DIR)).load_data()
    print("Documents loaded:", len(docs))
    for d in docs:
        print("File:", d.metadata.get("file_name"))

    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))

    return index


def search_docs(query_engine, query):
    print("\nDOC SEARCH")
    response = query_engine.query(query)

    print("\nRAW ANSWER")
    print(str(response))

    print("\nTOP CHUNKS USED")

    chunks = []
    if hasattr(response, "source_nodes"):
        for i, node in enumerate(response.source_nodes):
            text = node.node.text
            score = node.score
            file = node.node.metadata.get("file_name", "unknown")

            chunks.append(text)

            print(f"\nChunk {i}")
            print("File:", file)
            print("Score:", score)
            print(text[:400])

    return str(response), chunks


def search_web(query):
    print("\nWEB SEARCH (stub)")
    result = f"Web result placeholder for: {query}"
    print(result)
    return result


def rank_sources(llm, query, sources):
    print("\nRANKING SOURCES")

    def score(chunk):
        try:
            prompt = f"""
Score relevance (0-1).

Question:
{query}

Text:
{chunk}

Return only number.
"""
            return float(llm.complete(prompt).text.strip())
        except:
            return 0.5

    for s in sources:
        s["score"] = score(s["content"])
        print("\nSOURCE:", s["source"])
        print("Score:", s["score"])

    return sorted(sources, key=lambda x: x["score"], reverse=True)


def synthesize(llm, query, ranked):
    context = "\n\n".join(
        f"[{r['source']} | score={r['score']:.2f}]\n{r['content']}"
        for r in ranked
    )

    print("\nFINAL CONTEXT SENT TO LLM")
    print(context)

    prompt = f"""
Answer using the context below.

Question:
{query}

Context:
{context}
"""

    result = llm.complete(prompt).text

    print("\nFINAL ANSWER")
    print(result)

    return result


def pipeline(llm, query, query_engine):
    docs, chunks = search_docs(query_engine, query)
    web = search_web(query)

    sources = [
        {"source": "docs", "content": docs},
        {"source": "web", "content": web},
    ]

    ranked = rank_sources(llm, query, sources)
    synthesize(llm, query, ranked)


def main():
    index = build_index()
    query_engine = index.as_query_engine(similarity_top_k=5)

    llm = Settings.llm

    print("\nREADY\n")

    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            break

        pipeline(llm, q, query_engine)


if __name__ == "__main__":
    main()

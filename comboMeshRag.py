import os
import re
import asyncio
from pathlib import Path

from meshcore import MeshCore, EventType

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

from tavily import TavilyClient


DATA_DIR = Path(__file__).parent / "data"
STORAGE_DIR = Path(__file__).parent / "storage"

FORCE_REBUILD = False
MAX_MSG_LEN = 180

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# -------------------------
# EMBEDDINGS
# -------------------------

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.text_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
)

Settings.llm = Ollama(
    model="tinyllama",
    request_timeout=120.0,
    context_window=2048,
    additional_kwargs={
        "num_gpu": 0,
        "num_predict": 200,
    },
)


# -------------------------
# INDEX
# -------------------------

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

    index.storage_context.persist(
        persist_dir=str(STORAGE_DIR)
    )

    return index


# -------------------------
# DOC SEARCH
# -------------------------

def search_docs(query_engine, query):
    print("\nDOC SEARCH")

    response = query_engine.query(query)

    print("\nRAW ANSWER")
    print(str(response))

    chunks = []

    if hasattr(response, "source_nodes"):
        print("\nTOP CHUNKS USED")

        for i, node in enumerate(response.source_nodes):
            text = node.node.text
            score = node.score
            file = node.node.metadata.get(
                "file_name",
                "unknown"
            )

            chunks.append(text)

            print(f"\nChunk {i}")
            print("File:", file)
            print("Score:", score)
            print(text[:400])

    return str(response), chunks


# -------------------------
# WEB SEARCH
# -------------------------

def search_web(query):
    print("\nWEB SEARCH")

    try:
        result = client.search(query=query)

        text = "\n\n".join(
            r["content"]
            for r in result["results"]
            if "content" in r
        )

        print(text[:500])

        return text

    except Exception as e:
        print("Web search error:", e)
        return ""


# -------------------------
# RANKING
# -------------------------

def rank_sources(llm, query, sources):
    print("\nRANKING SOURCES")

    def score(chunk):
        try:
            prompt = f"""
Rate relevance from 0.0 to 1.0.

Question:
{query}

Text:
{chunk[:2000]}

Return ONLY a decimal number.
"""

            response = llm.complete(prompt).text.strip()

            print("RAW SCORE:", repr(response))

            match = re.search(
                r"\d*\.?\d+",
                response
            )

            if match:
                return float(match.group())

            return 0.5

        except Exception as e:
            print("Ranking error:", e)
            return 0.5

    for s in sources:
        s["score"] = score(s["content"])

        print(
            f"SOURCE={s['source']} SCORE={s['score']}"
        )

    return sorted(
        sources,
        key=lambda x: x["score"],
        reverse=True,
    )


# -------------------------
# SYNTHESIS
# -------------------------

def synthesize(llm, query, ranked):
    context = "\n\n".join(
        f"[{r['source']} | score={r['score']:.2f}]\n{r['content']}"
        for r in ranked
    )

    print("\nFINAL CONTEXT")
    print(context[:2000])

    prompt = f"""
Answer using the context below.

Requirements:
- Keep answer concise.
- Maximum 2 short paragraphs.
- Prefer under 300 characters.
- If unsure, say so.

Question:
{query}

Context:
{context}
"""

    result = llm.complete(prompt).text

    print("\nFINAL ANSWER")
    print(result)

    return result


# -------------------------
# PIPELINE
# -------------------------

def pipeline(llm, query, query_engine):
    docs_answer, chunks = search_docs(
        query_engine,
        query
    )

    web = search_web(query)

    sources = []

    for i, chunk in enumerate(chunks):
        sources.append(
            {
                "source": f"chunk_{i}",
                "content": chunk,
            }
        )

    if web:
        sources.append(
            {
                "source": "web",
                "content": web,
            }
        )

    ranked = rank_sources(
        llm,
        query,
        sources
    )

    ranked = ranked[:5]

    return synthesize(
        llm,
        query,
        ranked
    )


# -------------------------
# MESSAGE SPLITTING
# -------------------------

async def send_long_message(
    meshcore,
    contact,
    text
):
    chunks = [
        text[i:i + MAX_MSG_LEN]
        for i in range(
            0,
            len(text),
            MAX_MSG_LEN
        )
    ]

    total = len(chunks)

    print(
        f"Sending {total} chunk(s)"
    )

    for i, chunk in enumerate(
        chunks,
        start=1
    ):
        if total > 1:
            chunk = f"[{i}/{total}] {chunk}"

        await meshcore.commands.send_msg(
            contact,
            chunk
        )

        await asyncio.sleep(1)


# -------------------------
# MAIN
# -------------------------

async def main():
    index = build_index()

    query_engine = index.as_query_engine(
        similarity_top_k=5
    )

    llm = Settings.llm

    print("Connecting...")

    meshcore = await MeshCore.create_serial(
        "/dev/ttyACM0"
    )

    print("Connected!")

    result = await meshcore.commands.get_contacts()

    if result.type == EventType.ERROR:
        print(
            f"Error getting contacts: {result.payload}"
        )
        return

    contacts = result.payload

    print(
        f"Found {len(contacts)} contacts"
    )

    if not contacts:
        print("No contacts found")
        return

    contact = next(
        iter(contacts.items())
    )[1]

    print(
        f"Using contact: {contact}"
    )

    async def on_message(event):
        try:
            if "text" in event.payload:
                print("\n=== NEW MESSAGE ===")

                received = event.payload["text"]

                print(
                    f"Message: {received}"
                )

                message = pipeline(
                    llm,
                    received,
                    query_engine
                )

                print(
                    "RAG response:",
                    message
                )

                print(
                    "Length:",
                    len(message)
                )

                await send_long_message(
                    meshcore,
                    contact,
                    message
                )

        except Exception as e:
            print(
                f"Listener error: {e}"
            )

    meshcore.subscribe(
        EventType.CONTACT_MSG_RECV,
        on_message
    )

    await meshcore.start_auto_message_fetching()

    print("\nListener started!")

    try:
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        await meshcore.stop_auto_message_fetching()

        await meshcore.disconnect()

        print("Disconnected.")


asyncio.run(main())

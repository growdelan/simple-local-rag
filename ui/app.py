# WYMAGANIA:
# streaming_response = query_engine.query(query_text)
# pip install "sentence-transformers>=3.0.0" chromadb llama-index gradio nest_asyncio
# (na macOS pamiętaj o torch z obsługą MPS, jeśli chcesz użyć "mps" dla rerankera)

import os
import shutil
import gradio as gr
import chromadb
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank


CUSTOM_CSS = """
.thinking-msg {
    color: #888;
    font-style: italic;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.thinking-msg .dots {
    display: inline-flex;
    gap: 4px;
    margin-left: 2px;
}

.thinking-msg .dots span {
    width: 6px;
    height: 6px;
    background-color: #bbb;
    border-radius: 50%;
    opacity: 0.25;
    animation: thinking-bounce 1.2s infinite ease-in-out;
}

.thinking-msg .dots span:nth-child(2) {
    animation-delay: 0.2s;
}

.thinking-msg .dots span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes thinking-bounce {
    0%, 80%, 100% {
        transform: translateY(0);
        opacity: 0.25;
    }
    40% {
        transform: translateY(-4px);
        opacity: 0.7;
    }
}
"""

# ========================
# KONFIGURACJA MODELI / PIPE
# ========================
STANDARD_MODEL = "gemma2:latest"
PRO_MODEL = "qwen3:4b"
QUESTION_MODEL = "gemma3:4b-it-qat"
EMBED_MODEL_NAME = "embeddinggemma:latest"

# Reranker cross-encoder (PL / wielojęzyczny)
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
# Dla okna 8192 lepiej nie przepychać do LLM zbyt wielu chunków naraz
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "8"))  # ile fragmentów trafi do LLM (po reranku)
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")  # "cpu" (bezpiecznie) lub "mps"

# Limity generatora (Ollama options)
# Uwaga: w llama.cpp kontekst (num_ctx) obejmuje też generację, więc zostawiamy miejsce na odpowiedź.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))  # sensowny zapas na odpowiedź

# Chunking pod książki: większe chunki + większy overlap, żeby nie rwać wątku w połowie
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Czy przy tworzeniu kolekcji kasować istniejącą (żeby uniknąć duplikatów)
RESET_COLLECTION_ON_CREATE = os.getenv("RESET_COLLECTION_ON_CREATE", "true").lower() in (
    "1",
    "true",
    "yes",
    "y",
)

# Parametry HNSW dla nowych kolekcji (zwiększają recall shortlisty)
HNSW_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:M": 32,
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": 200,
}

# Które metadane zostawiamy dla LLM (dla cytowania/źródła)
KEEP_LLM_METADATA_KEYS = {
    "file_name",
    "page_label",
    "page_number",
    "source",
    "title",
}

# ================
# PROMPTY / TEMPLATES
# ================
NODE_TEMPLATE = """
Kontekst: {context_str}.
Podaj tytuł, który podsumowuje wszystkie unikalne jednostki, nazwy własne lub motywy występujące w kontekście. Podaj tytuł nie dodawaj nic więcej. Użyj języka polskiego.
Tytuł:
"""

COMBINE_TEMPLATE = """
Użyj języka polskiego.
Kontekst: {context_str}
Na podstawie powyższych propozycji tytułów oraz treści, jaki będzie najbardziej trafny i całościowy tytuł tego dokumentu? Podaj sam tytuł, nie dodawaj nic więcej (to bardzo ważne!!!).
Tytuł:
"""

QUESTION_TEMPLATE = """
Oto kontekst:
{context_str}

Na podstawie powyższego kontekstu, wygeneruj {num_questions} pytań, na które ten kontekst może dostarczyć konkretnych odpowiedzi, trudnych do znalezienia w innych źródeł.

Można również uwzględnić podsumowania szerszego kontekstu. Postaraj się wykorzystać te podsumowania, aby stworzyć lepsze pytania, na które niniejszy kontekst może odpowiedzieć.
W odpowiedzi podaj same pytania, nie dodawaj nic więcej.
Użyj języka polskiego.
"""

RAG_SYSTEM_PROMPT = """
Jesteś asystentem RAG. Odpowiadasz tylko na podstawie przekazanego kontekstu.
Nie dodawaj nic spoza niego. Jeśli brak odpowiedzi — napisz: "Nie znaleziono w dostarczonym kontekście."

Odpowiadaj po polsku, zwięźle i precyzyjnie.
Podaj cytat potwierdzający (1–2 zdania) i źródło (np. nazwa pliku i strona), jeśli jest.

Format:
Odpowiedź: ...
Uzasadnienie: "..."
Źródło: ...
"""

text_qa_template = PromptTemplate(
    "Użyj języka polskiego.\n"
    "Kontekst:\n{context_str}\n\n"
    "Pytanie:\n{query_str}\n\n"
    "Odpowiedz zgodnie z zasadami systemowymi."
)

# ========================
# GLOBALNY RERANKER (ładuje się raz, oszczędza narzut)
# ========================
_global_rerank = SentenceTransformerRerank(
    model=RERANK_MODEL_NAME,
    top_n=RERANK_TOP_N,
    device=RERANK_DEVICE,
    cross_encoder_kwargs={"max_length": 1024},
)

# ========================
# POMOCNICZE: CHROMA + METADATA
# ========================
def _get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path="./chroma_db")


def _collection_exists(client: chromadb.PersistentClient, name: str) -> bool:
    try:
        cols = client.list_collections()
        return any(getattr(c, "name", None) == name for c in cols)
    except Exception:
        return False


def _get_or_create_collection_with_metadata(client: chromadb.PersistentClient, name: str):
    # get_or_create_collection nie aktualizuje metadata istniejącej kolekcji.
    # Tutaj tworzymy nową z HNSW_METADATA tylko gdy nie istnieje.
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name, metadata=HNSW_METADATA)


def _sanitize_docs_metadata(docs):
    """
    Cel:
    - embeddingi mają być "czyste" (bez metadanych doklejanych do treści),
    - LLM ma widzieć tylko sensowne źródła (file_name/page_label),
    - usuwamy ryzyko, że retrieval będzie "o metadanych" zamiast o treści.
    """
    for doc in docs:
        # Nie doklejamy metadanych do tekstu (embedding ma bazować na treści)
        # doc.text_template = ...  <-- USUNIĘTE

        # Ustandaryzuj źródło
        meta = getattr(doc, "metadata", {}) or {}
        file_name = meta.get("file_name") or meta.get("filename") or meta.get("source") or ""
        if file_name:
            meta["source"] = file_name
        doc.metadata = meta

        # Embedding: wyklucz wszystkie metadane
        try:
            doc.excluded_embed_metadata_keys = list(doc.metadata.keys())
        except Exception:
            pass

        # LLM: wyklucz hałas, zostaw tylko przydatne do cytowania
        try:
            doc.excluded_llm_metadata_keys = [
                k for k in doc.metadata.keys() if k not in KEEP_LLM_METADATA_KEYS
            ]
        except Exception:
            pass

    return docs


# ========================
# FUNKCJE APLIKACJI
# ========================
def get_collection_names():
    try:
        client = _get_chroma_client()
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception:
        return []


def create_collection(files, collection_name, pro_embeddings=False):
    if not collection_name:
        return gr.update(), "Error: collection name is required."
    if not files:
        return gr.update(), "Error: no files uploaded."

    # Prepare data directory
    tmp_dir = os.path.join("./data", collection_name)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    saved_paths = []
    for file_path in files:
        try:
            dest = os.path.join(tmp_dir, os.path.basename(file_path))
            shutil.copy(file_path, dest)
            saved_paths.append(dest)
        except Exception:
            continue

    if not saved_paths:
        return gr.update(), "Error: no valid files to process."

    try:
        # Load documents
        docs = SimpleDirectoryReader(input_dir=tmp_dir).load_data()
        docs = _sanitize_docs_metadata(docs)

        # Embedding model (Ollama)
        embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)

        # Splitter pod książki: większe chunki, overlap, zachowaj akapity
        text_splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            # separator zostawiamy jako spację do składania zdań,
            # ale paragraph_separator dba o granice akapitów
            separator=" ",
            paragraph_separator="\n\n",
        )

        # Choose pipeline based on "Pro" flag (to jest enrichment, nie inne embeddingi)
        if pro_embeddings:
            llm = Ollama(
                model=QUESTION_MODEL,
                request_timeout=300.0,
                temperature=1,
                context_window=OLLAMA_NUM_CTX,
                json_mode=False,
                additional_kwargs={
                    "num_ctx": OLLAMA_NUM_CTX,
                    "num_predict": 256,  # ingestion nie potrzebuje długich generacji
                },
            )
            title_extractor = TitleExtractor(
                llm=llm,
                nodes=5,
                node_template=NODE_TEMPLATE,
                combine_template=COMBINE_TEMPLATE,
            )
            qa_extractor = QuestionsAnsweredExtractor(
                llm=llm, questions=3, prompt_template=QUESTION_TEMPLATE
            )
            pipeline = IngestionPipeline(
                transformations=[
                    text_splitter,
                    title_extractor,
                    qa_extractor,
                ]
            )
        else:
            pipeline = IngestionPipeline(transformations=[text_splitter])

        # Run ingestion
        import asyncio
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        import nest_asyncio
        nest_asyncio.apply()

        nodes = pipeline.run(
            documents=docs,
            in_place=True,
            show_progress=True,
        )

        # Debug: zapis chunków per kolekcja (żeby nie nadpisywać wszystkiego w output.txt)
        debug_path = os.path.join("./data", collection_name, "debug_chunks.txt")
        with open(debug_path, "w", encoding="utf-8") as f:
            for idx, node in enumerate(nodes):
                try:
                    content = node.get_content(metadata_mode=MetadataMode.LLM)
                except Exception:
                    content = node.get_content()
                f.write(f"Chunk {idx}\n")
                f.write(content)
                f.write("\n\n")

        # Persist to ChromaDB
        client = _get_chroma_client()

        # Najważniejsze: unikamy duplikatów w Chroma przy ponownym create
        if RESET_COLLECTION_ON_CREATE and _collection_exists(client, collection_name):
            try:
                client.delete_collection(collection_name)
            except Exception:
                # jeśli delete nie przejdzie (np. race), spróbujemy dalej get_or_create
                pass

        # Tworzymy/otwieramy kolekcję; jeśli nowa, dostaje HNSW metadata
        chroma_collection = _get_or_create_collection_with_metadata(client, collection_name)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Budowa indeksu zapisze wektory do Chroma
        VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
        )

    except Exception as e:
        return gr.update(), f"Error creating collection: {e}"

    new_choices = get_collection_names()
    return gr.update(
        choices=new_choices, value=collection_name
    ), f"Collection {collection_name} created successfully. (chunks: {len(nodes)})"


def delete_collection(collection_name):
    if not collection_name:
        return gr.update(), "Error: select a collection to delete."
    try:
        client = _get_chroma_client()
        client.delete_collection(collection_name)
        shutil.rmtree(os.path.join("./data", collection_name), ignore_errors=True)
    except Exception as e:
        return gr.update(), f"Error deleting collection: {e}"
    new_choices = get_collection_names()
    return gr.update(
        choices=new_choices, value=None
    ), f"Collection {collection_name} deleted."


def parse_reasoning_and_answer(text):
    """
    Parse the model's response to extract reasoning and final answer.
    Expected format: <think>reasoning</think> final answer
    """
    import re

    reasoning, answer = "", text
    think_match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        answer = think_match.group(2).strip()
    return reasoning, answer


def query_collection(
    collection_name, query_text, history, use_reasoning=False, use_rerank=True
):
    history = history or []
    if not collection_name:
        yield history, ""
        return

    # append user message
    history.append({"role": "user", "content": query_text})
    yield history, ""

    # placeholder: "model pracuje"
    model_response = ""
    thinking_msg = (
        '<span class="thinking-msg">Model przygotowuje odpowiedź'
        '<span class="dots"><span></span><span></span><span></span></span>'
        "</span>"
    )
    history.append({"role": "assistant", "content": thinking_msg})
    yield history, ""

    try:
        model_name = PRO_MODEL if use_reasoning else STANDARD_MODEL

        # Dla PRO (reasoning) zostawiamy większą swobodę, ale nadal ograniczamy,
        # bo num_ctx=8192 obejmuje też generację.
        # Jeśli chcesz "dłużej", zwiększ OLLAMA_NUM_PREDICT, ale pilnuj kontekstu.
        num_predict = -1 if (model_name == PRO_MODEL and OLLAMA_NUM_PREDICT <= 0) else OLLAMA_NUM_PREDICT

        llm = Ollama(
            model=model_name,
            request_timeout=300.0,
            temperature=0.25,
            context_window=OLLAMA_NUM_CTX,
            json_mode=False,
            additional_kwargs={
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": num_predict,
            },
            system_prompt=RAG_SYSTEM_PROMPT,
        )

        # Embedding model (Ollama) - musi być ten sam co w indeksowaniu
        embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)

        # Vector store
        client = _get_chroma_client()
        chroma_collection = _get_or_create_collection_with_metadata(client, collection_name)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

        # similarity_top_k: shortlist pod rerank
        q_len = len(query_text.split())
        if use_rerank:
            # umiarkowanie duży shortlist; przy książkach i tak robi robotę reranker
            sim_k = 55 if q_len > 6 else 40
            postprocessors = [_global_rerank]
        else:
            sim_k = 12
            postprocessors = None

        query_engine = index.as_query_engine(
            llm=llm,
            streaming=True,
            similarity_top_k=sim_k,
            node_postprocessors=postprocessors,
            text_qa_template=text_qa_template,
        )

        streaming_response = query_engine.query(query_text)

        # Debug: pokaż kontekst przekazywany do LLM
        source_nodes = getattr(streaming_response, "source_nodes", None)
        if source_nodes:
            print(
                "\n=== Kontekst przekazywany do LLM (rerank "
                f"{'ON' if use_rerank else 'OFF'}) ==="
            )
            for idx, node_with_score in enumerate(source_nodes, start=1):
                node = getattr(node_with_score, "node", None)
                score = getattr(node_with_score, "score", None)
                score_repr = (
                    f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
                )
                if node is None:
                    print(f"[{idx}] Brak danych węzła (score: {score_repr})")
                    continue
                try:
                    content = node.get_content(metadata_mode=MetadataMode.LLM)
                except Exception:
                    content = node.get_content()
                print(f"[{idx}] Score: {score_repr}")
                print(content)
                print("---")
            print("=== Koniec kontekstu ===\n")

        # yield tokens as they arrive
        for text in streaming_response.response_gen:
            model_response += str(text)
            history[-1]["content"] = model_response
            yield history, ""

        # After streaming: optional reasoning block
        reasoning, answer = parse_reasoning_and_answer(model_response)
        if reasoning:
            reasoning_md = (
                "<details><summary><b>Myślenie modelu (kliknij, aby rozwinąć)</b></summary>\n\n"
                f"{reasoning}\n</details>"
            )
            model_response = f"{reasoning_md}\n\n{answer}"

        history[-1]["content"] = model_response
        yield history, ""

    except Exception as e:
        error_msg = f"Error: {e}"
        history[-1]["content"] = error_msg
        yield history, ""
        return


def clear_chat(_):
    return []


def format_chatbot_message(message):
    if isinstance(message, str):
        return message
    elif isinstance(message, dict) and "content" in message:
        content = message["content"]
        reasoning, answer = parse_reasoning_and_answer(content)
        if reasoning:
            return f"""
            <details style="margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px; padding: 8px 12px;">
                <summary style="cursor: pointer; font-weight: bold; color: #666;">My Thinking Process (click to expand)</summary>
                <div style="margin-top: 8px; padding: 8px; background: #f8f9fa; border-radius: 4px; white-space: pre-wrap;">
                    {reasoning}
                </div>
            </details>
            <div style="margin-top: 12px;">
                {answer}
            </div>
            """
        return answer
    return str(message)


def main():
    with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
        gr.Markdown("# RAG Chatbot UI")

        with gr.Row():
            with gr.Column(scale=3):
                collection_dropdown = gr.Dropdown(
                    choices=get_collection_names(),
                    label="Select Collection",
                    value=None,
                )
                delete_btn = gr.Button("Delete Collection")
                refresh_btn = gr.Button("Refresh Collections")

                reasoning_checkbox = gr.Checkbox(
                    label="Reasoning",
                    value=False,
                    info="Użyj zaawansowanego modelu Qwen3 do odpowiedzi z reasoningiem.",
                )
                rerank_checkbox = gr.Checkbox(
                    label="Rerank (SentenceTransformer)",
                    value=True,
                    info="Odznacz, aby pominąć reranker i użyć surowego wyniku wektorowego.",
                )

                chatbot = gr.Chatbot(label="Chat", height=500, type="messages")
                msg_input = gr.Textbox(
                    label="Your message",
                    placeholder="Type your question here...",
                )
                send_btn = gr.Button("Send")

            with gr.Column(scale=1):
                gr.Markdown("## Create New Collection")
                new_collection_name = gr.Textbox(label="Collection Name")
                file_uploader = gr.File(
                    file_count="multiple",
                    type="filepath",
                    label="Upload Files",
                )
                pro_checkbox = gr.Checkbox(
                    label="Pro (Enrichment: Title + Q&A)",
                    value=False,
                    info="Dodaje tytuł i pytania/odpowiedzi jako enrichment. To nie są inne embeddingi.",
                )
                create_btn = gr.Button("Create Collection")
                status_output = gr.Textbox(label="Status")

        create_btn.click(
            create_collection,
            inputs=[file_uploader, new_collection_name, pro_checkbox],
            outputs=[collection_dropdown, status_output],
        )
        delete_btn.click(
            delete_collection,
            inputs=[collection_dropdown],
            outputs=[collection_dropdown, status_output],
        )

        def refresh_collections():
            return gr.update(choices=get_collection_names())

        refresh_btn.click(refresh_collections, inputs=[], outputs=[collection_dropdown])

        collection_dropdown.change(
            clear_chat, inputs=[collection_dropdown], outputs=[chatbot]
        )

        send_btn.click(
            query_collection,
            inputs=[
                collection_dropdown,
                msg_input,
                chatbot,
                reasoning_checkbox,
                rerank_checkbox,
            ],
            outputs=[chatbot, msg_input],
        )
        msg_input.submit(
            query_collection,
            inputs=[
                collection_dropdown,
                msg_input,
                chatbot,
                reasoning_checkbox,
                rerank_checkbox,
            ],
            outputs=[chatbot, msg_input],
        )

        demo.launch()


if __name__ == "__main__":
    main()

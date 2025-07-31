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
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor

STANDARD_MODEL = 'gemma3:4b-it-qat'
PRO_MODEL = 'qwen3:4b'

# Templates for advanced Pro Embeddings pipeline
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


def get_collection_names():
    try:
        client = chromadb.PersistentClient(path='./chroma_db')
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception:
        return []


def create_collection(files, collection_name, pro_embeddings=False):
    if not collection_name:
        return gr.update(), 'Error: collection name is required.'
    if not files:
        return gr.update(), 'Error: no files uploaded.'

    # Prepare data directory
    tmp_dir = os.path.join('./data', collection_name)
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
        return gr.update(), 'Error: no valid files to process.'

    try:
        # Load documents
        docs = SimpleDirectoryReader(input_dir=tmp_dir).load_data()
        for doc in docs:
            doc.text_template = 'Metadata:\n{metadata_str}\n---\nContent:\n{content}'
            if 'page_label' not in doc.excluded_embed_metadata_keys:
                doc.excluded_embed_metadata_keys.append('page_label')

        # Choose pipeline based on Pro Embeddings flag
        if pro_embeddings:
            # Advanced pipeline
            llm = Ollama(
                model=STANDARD_MODEL,
                request_timeout=300.0,
                temperature=0.7,
                context_window=8192,
                json_mode=False,
            )
            embed_model = OllamaEmbedding(model_name='nomic-embed-text')
            text_splitter = SentenceSplitter(
                separator=' ', chunk_size=1024, chunk_overlap=128
            )
            title_extractor = TitleExtractor(
                llm=llm, nodes=5,
                node_template=NODE_TEMPLATE,
                combine_template=COMBINE_TEMPLATE
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
            # Standard pipeline
            embed_model = OllamaEmbedding(model_name='nomic-embed-text')
            text_splitter = SentenceSplitter(
                separator=' ', chunk_size=1024, chunk_overlap=128
            )
            pipeline = IngestionPipeline(
                transformations=[text_splitter]
            )

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
        # Save nodes to a text file
        with open('output.txt', 'w', encoding='utf-8') as f:
            for idx, node in enumerate(nodes):
                content = node.get_content(metadata_mode=MetadataMode.LLM)
                f.write(f"Chunk {idx}\n")
                f.write(content)
                f.write("\n\n")
        # Persist to ChromaDB
        client = chromadb.PersistentClient(path='./chroma_db')
        chroma_collection = client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

    except Exception as e:
        return gr.update(), f'Error creating collection: {e}'

    new_choices = get_collection_names()
    return gr.update(choices=new_choices, value=collection_name), f'Collection {collection_name} created successfully.'


def delete_collection(collection_name):
    if not collection_name:
        return gr.update(), 'Error: select a collection to delete.'
    try:
        client = chromadb.PersistentClient(path='./chroma_db')
        client.delete_collection(collection_name)
        shutil.rmtree(os.path.join('./data', collection_name), ignore_errors=True)
    except Exception as e:
        return gr.update(), f'Error deleting collection: {e}'
    new_choices = get_collection_names()
    return gr.update(choices=new_choices, value=None), f'Collection {collection_name} deleted.'


def parse_reasoning_and_answer(text):
    """
    Parse the model's response to extract reasoning and final answer.
    Expected format: <think>reasoning</think> final answer
    """
    import re
    reasoning, answer = '', text
    # Look for <think> tags
    think_match = re.search(r'<think>(.*?)</think>(.*)', text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        answer = think_match.group(2).strip()
    return reasoning, answer

def query_collection(collection_name, query_text, history, use_reasoning=False):
    history = history or []
    if not collection_name:
        yield history, ''
        return
    # append user message
    history.append({"role": "user", "content": query_text})
    # placeholder for streaming assistant response
    model_response = ""
    history.append({"role": "assistant", "content": model_response})
    try:
        model_name = PRO_MODEL if use_reasoning else STANDARD_MODEL
        llm = Ollama(
            model=model_name,
            request_timeout=300.0,
            temperature=0.7,
            context_window=8192,
            json_mode=False,
        )
        embed_model = OllamaEmbedding(model_name='nomic-embed-text')
        client = chromadb.PersistentClient(path='./chroma_db')
        chroma_collection = client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        # enable streaming in the query engine
        query_engine = index.as_query_engine(llm=llm, streaming=True, similarity_top_k=5)
        streaming_response = query_engine.query(query_text)
        # yield tokens as they arrive
        for text in streaming_response.response_gen:
            model_response += str(text)
            # Update the assistant's response in history
            history[-1]["content"] = model_response
            yield history, ''
        # After streaming, parse reasoning and answer and build markdown
        reasoning, answer = parse_reasoning_and_answer(model_response)
        if reasoning:
            reasoning_md = f"<details><summary><b>Myślenie modelu (kliknij, aby rozwinąć)</b></summary>\n\n{reasoning}\n</details>"
            model_response = f"{reasoning_md}\n\n{answer}"
        # Update the final response in history
        history[-1]["content"] = model_response
        yield history, ''
    except Exception as e:
        error_msg = f'Error: {e}'
        history[-1]["content"] = error_msg
        yield history, ''
        return



def clear_chat(_):
    return []


def format_chatbot_message(message):
    if isinstance(message, str):
        return message
    elif isinstance(message, dict) and 'content' in message:
        content = message['content']
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
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown('# RAG Chatbot UI')
        with gr.Row():
            with gr.Column(scale=3):
                collection_dropdown = gr.Dropdown(
                    choices=get_collection_names(), label='Select Collection', value=None
                )
                delete_btn = gr.Button('Delete Collection')
                # Button to refresh the list of collections without restarting the server
                refresh_btn = gr.Button('Refresh Collections')
                reasoning_checkbox = gr.Checkbox(
                    label='Reasoning', value=False,
                    info='Użyj zaawansowanego modelu Qwen3 do odpowiedzi z reasoningiem.'
                )
                chatbot = gr.Chatbot(label='Chat', height=500, type='messages')
                msg_input = gr.Textbox(
                    label='Your message', placeholder='Type your question here...'
                )
                send_btn = gr.Button('Send')
            with gr.Column(scale=1):
                gr.Markdown('## Create New Collection')
                new_collection_name = gr.Textbox(label='Collection Name')
                file_uploader = gr.File(
                    file_count='multiple', type='filepath', label='Upload Files'
                )
                pro_checkbox = gr.Checkbox(label='Pro Embeddings', value=False)
                create_btn = gr.Button('Create Collection')
                status_output = gr.Textbox(label='Status')

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
        # Wire up the refresh button to update the dropdown choices
        def refresh_collections():
            return gr.update(choices=get_collection_names())

        refresh_btn.click(
            refresh_collections,
            inputs=[],
            outputs=[collection_dropdown]
        )
        collection_dropdown.change(
            clear_chat, inputs=[collection_dropdown], outputs=[chatbot]
        )
        send_btn.click(
            query_collection,
            inputs=[collection_dropdown, msg_input, chatbot, reasoning_checkbox],
            outputs=[chatbot, msg_input],
        )
        msg_input.submit(
            query_collection,
            inputs=[collection_dropdown, msg_input, chatbot, reasoning_checkbox],
            outputs=[chatbot, msg_input],
        )
        demo.launch()

if __name__ == '__main__':
    main()

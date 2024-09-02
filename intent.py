import os
import time
import markdown
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Any
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def process_batch(vectorstore, batch: List[Any]) -> str:
    try:
        vectorstore.add_documents(documents=batch)
        return f"Processed batch of size {len(batch)}"
    except Exception as e:
        return f"Error processing batch: {e}"

def get_optimal_thread_count() -> int:
    num_cores = os.cpu_count() - 2  # Leaving 2 CPUs idle
    return num_cores * 2  # Typically, 2x the number of cores for I/O-bound tasks

def add_chunks_to_vectorstore(vectorstore, chunks: List[Any], batch_size: int = 1000) -> List[str]:
    messages = []
    total_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size else 0)
    num_threads = get_optimal_thread_count()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_batch = {executor.submit(process_batch, vectorstore, batch): batch for batch in chunk_list(chunks, batch_size)}
        
        for i, future in enumerate(tqdm(as_completed(future_to_batch), total=total_batches, desc="Processing Batches")):
            try:
                result = future.result()
            except Exception as e:
                messages.append(f"Error retrieving result: {e}")

    return messages

def ingest_md_files_to_chroma(folder_path: str, base_persist_directory: str) -> List[str]:
    messages = []

    # Set persistence directory dynamically based on the folder name, outside of `docs`
    folder_name = os.path.basename(folder_path)
    persist_directory = os.path.join(base_persist_directory, folder_name)

    embedding_model = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text:v1.5')  # Default value

    local_embeddings = OllamaEmbeddings(model=embedding_model)
    
    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=local_embeddings)
        messages.append(f"Loaded the existing Chroma DB for {persist_directory}")
    else:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=local_embeddings)
        messages.append(f"Created the Chroma DB for {persist_directory}")
    
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.md'):
            with open(file_path, 'r', encoding='utf8') as file:
                content = file.read()
                # Convert Markdown to HTML
                html_content = markdown.markdown(content)
                # Use BeautifulSoup to strip HTML tags
                soup = BeautifulSoup(html_content, 'html.parser')
                plain_text = soup.get_text().replace('\n', ' ').strip()
                doc = Document(page_content=plain_text, metadata={"filename": filename,"version":folder_name, "type": "laravel-docs",})
                documents.append(doc)
    
    if not documents:
        messages.append(f"No Markdown files found in {folder_path}")
        return messages

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(documents)
    print(f"Number of splits for {persist_directory}: {len(all_splits)}")

    print(f"Adding vectors to the DB for {persist_directory} started")
    batch_size = 100
    start_time = time.time()
    status_messages = add_chunks_to_vectorstore(vectorstore, all_splits, batch_size)
    elapsed_time = (time.time() - start_time) / 60
    messages.extend(status_messages)
    messages.append(f"Added Markdown data to Chroma DB for {persist_directory} in {elapsed_time:.2f} minutes")
    
    return messages

def process_folder(folder_path: str, base_persist_directory: str) -> List[str]:
    messages = []
    
    # Iterate through the current folder
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            # Check if the subfolder contains any Markdown files or subfolders
            if any(filename.endswith('.md') for filename in os.listdir(subfolder_path)) or any(os.path.isdir(os.path.join(subfolder_path, d)) for d in os.listdir(subfolder_path)):
                # Process the subfolder
                messages.extend(process_folder(subfolder_path, base_persist_directory))  # Recursively process subfolders
                print(f"Processing folder {subfolder_path}")
                md_messages = ingest_md_files_to_chroma(subfolder_path, base_persist_directory)
                messages.extend(md_messages)
            else:
                messages.append(f"No Markdown files found in {subfolder_path}")
    
    return messages

# Example usage
parent_folder = 'docs'  # Parent folder containing subdirectories
base_persist_directory = os.getenv('PERSIST_DIRECTORY', 'db')  # Base directory outside of `docs` for Chroma DB
print(f"Processing parent folder {parent_folder}")
parent_messages = process_folder(parent_folder, base_persist_directory)
for message in parent_messages:
    print(message)

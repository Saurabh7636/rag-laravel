from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

def format_docs(docs):
    return "\n\n".join(f"Sentence: {doc.page_content}\nIntent: {doc.metadata.get('intent', 'Unknown')}" for doc in docs)

def detect_intent_with_context(sentence:str,version:str):
    messages = []

    embedding_model = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text:v1.5')  # Default value
    persist_directory = os.getenv('PERSIST_DIRECTORY', 'db')  # Default value

    local_embeddings = OllamaEmbeddings(model=embedding_model)
    
    vectorstore = Chroma(persist_directory=f"{persist_directory}/{version}", embedding_function=local_embeddings)
    
    docs = vectorstore.similarity_search(sentence)
    if not docs:
        messages.append("No relevant intent was found.")
        return {"response": "No matching intent found.", "messages": messages}
    
    # Define the RAG prompt template for intent detection
    INTENT_TEMPLATE = """
    You are a documentation guide to the all laravel developer. Use the following pieces of retrieved context to determine the possible answer for the asked questions. If you cannot determine the answer, just say that you don't know. 

    <context>
    {context}
    </context>

    What is the answer of the following question? Just reply with the exact answer and some code example explanation and nothing else as response.

    {sentence}"""
    
    intent_prompt = ChatPromptTemplate.from_template(INTENT_TEMPLATE)

    chat_model = os.getenv('CHAT_MDOEL', 'llama3.1:8b')  # Default value
    model = ChatOllama(model=chat_model)
    
    chain = (
        RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
        | intent_prompt
        | model
        | StrOutputParser()
    )
    
    response = chain.invoke({"context": docs, "sentence": sentence})
    return response

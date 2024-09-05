import openai
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
import streamlit as st

# Fetch API Key from Streamlit secrets
api_key = st.secrets["openai"]["api_key"]
if not api_key:
    raise ValueError("API key not found in Streamlit secrets")

openai.api_key = api_key
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_store = None

def initialize_data_from_dataset(dataset):
    global vector_store

    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("Expected a pandas DataFrame, got a different data type.")

    rows_as_text = dataset.apply(lambda row: row.to_json(), axis=1).tolist()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = [chunk for text in rows_as_text for chunk in text_splitter.split_text(text)]
    documents = [Document(page_content=text, metadata={}) for text in texts]

    vector_store = FAISS.from_documents(documents, embeddings)
    print("Vector store initialized with dataset documents")

def get_chatbot_response(user_input, dataset, conversation_history):
    global vector_store
    if vector_store is None:
        print("Initializing vector store with dataset")
        initialize_data_from_dataset(dataset)

    retrieved_docs = vector_store.similarity_search(user_input, k=10)  # Adjust k as needed
    qa_chain = load_qa_chain(llm=ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=api_key))
    conversation_history.add_message({"role": "user", "content": user_input})
    print(f"User input: {user_input}")

    try:
        response = qa_chain.invoke({"question": user_input, "input_documents": retrieved_docs})
        print(f"Chatbot response: {response}")
    except Exception as e:
        print(f"Error generating response: {e}")
        response = {"output_text": "There was an error processing your request."}

    conversation_history.add_message({"role": "assistant", "content": response.get('output_text', 'No response content found.').strip()})
    return response.get('output_text', 'No response content found.').strip()

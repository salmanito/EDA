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
api_key = st.secrets.get("openai", {}).get("api_key", "")
if not api_key:
    st.error("API key not found in Streamlit secrets.")
    raise ValueError("API key not found in Streamlit secrets.")

# Initialize OpenAI API
openai.api_key = api_key
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_store = None

def initialize_data_from_dataset(dataset):
    global vector_store

    # Ensure dataset is a DataFrame
    if not isinstance(dataset, pd.DataFrame):
        st.error("Expected a pandas DataFrame, got a different data type.")
        raise ValueError("Expected a pandas DataFrame, got a different data type.")

    # Convert dataset to a list of strings (rows)
    rows_as_text = dataset.apply(lambda row: row.to_json(), axis=1).tolist()

    # Split text into chunks if needed
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = [chunk for text in rows_as_text for chunk in text_splitter.split_text(text)]

    # Create Document objects with metadata if needed
    documents = [Document(page_content=text, metadata={}) for text in texts]

    # Store documents in vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    st.info("Vector store initialized with dataset documents")

def get_chatbot_response(user_input, dataset, conversation_history):
    global vector_store
    if vector_store is None:
        st.info("Initializing vector store with dataset")
        initialize_data_from_dataset(dataset)

    # Retrieve a reasonable number of documents
    try:
        retrieved_docs = vector_store.similarity_search(user_input, k=10)  # Adjust k as needed
        qa_chain = load_qa_chain(llm=ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=api_key))
        conversation_history.add_message({"role": "user", "content": user_input})
        st.write(f"User input: {user_input}")

        response = qa_chain.invoke({"question": user_input, "input_documents": retrieved_docs})
        st.write(f"Chatbot response: {response}")

        conversation_history.add_message({"role": "assistant", "content": response.get('output_text', 'No response content found.').strip()})
        return response.get('output_text', 'No response content found.').strip()
    
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "There was an error processing your request."

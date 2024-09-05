import openai
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Updated import due to deprecation
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

# API Key and Embeddings Initialization
api_key = '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'
openai.api_key = api_key
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_store = None

def initialize_data_from_dataset(dataset):
    global vector_store

    # Ensure dataset is a DataFrame
    if not isinstance(dataset, pd.DataFrame):
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
    print("Vector store initialized with dataset documents")

def get_chatbot_response(user_input, dataset, conversation_history):
    global vector_store
    if vector_store is None:
        print("Initializing vector store with dataset")
        initialize_data_from_dataset(dataset)

    retrieved_docs = vector_store.similarity_search(user_input, k=100000)
    qa_chain = load_qa_chain(llm=ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=api_key))
    conversation_history.add_message({"role": "user", "content": user_input})
    print(f"User input: {user_input}")

    try:
        response = qa_chain.invoke({"question": user_input, "input_documents": retrieved_docs})
        print(f"Chatbot response: {response}")
    except Exception as e:
        print(f"Error generating response: {e}")
        response = {"output_text": "There was an error processing your request."}

    conversation_history.add_message({"role": "assistant", "content": response['output_text']})
    return response.get('output_text', 'No response content found.').strip()

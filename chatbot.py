import openai
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Updated import due to deprecation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain_openai import ChatOpenAI

# API Key and Embeddings Initialization
api_key = 'your-openai-api-key'
openai.api_key = api_key
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_store = None
tfidf_vectorizer = None
tfidf_matrix = None
documents = []

def initialize_data_from_dataset(dataset):
    global vector_store, tfidf_vectorizer, tfidf_matrix, documents

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

    # Create TF-IDF vectorizer and matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([doc.page_content for doc in documents])
    print("TF-IDF matrix initialized with dataset documents")

def get_chatbot_response(user_input, dataset, conversation_history):
    global vector_store, tfidf_vectorizer, tfidf_matrix, documents
    if vector_store is None:
        print("Initializing data from dataset")
        initialize_data_from_dataset(dataset)

    # Vectorize user input
    user_input_vector = tfidf_vectorizer.transform([user_input])

    # Compute similarity scores
    similarity_scores = cosine_similarity(user_input_vector, tfidf_matrix).flatten()

    # Retrieve the most similar document
    most_similar_index = similarity_scores.argmax()
    retrieved_doc = documents[most_similar_index]

    qa_chain = load_qa_chain(llm=ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=api_key))
    conversation_history.add_message({"role": "user", "content": user_input})
    print(f"User input: {user_input}")

    try:
        response = qa_chain.invoke({"question": user_input, "input_documents": [retrieved_doc]})
        print(f"Chatbot response: {response}")
    except Exception as e:
        print(f"Error generating response: {e}")
        response = {"output_text": "There was an error processing your request."}

    conversation_history.add_message({"role": "assistant", "content": response['output_text']})
    return response.get('output_text', 'No response content found.').strip()

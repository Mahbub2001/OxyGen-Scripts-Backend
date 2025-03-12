import os
import time
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

data_folder = "./data"
chunk_size = 1000
chunk_overlap = 50
check_interval = 10
pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
index_name = "oxygen"
index = pc.Index(index_name)

def batch_upsert(index, to_upsert, batch_size=100):
    """Splits the data into smaller batches to avoid Pinecone's 4MB limit."""
    for i in range(0, len(to_upsert), batch_size):
        batch = to_upsert[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i//batch_size + 1}/{(len(to_upsert) // batch_size) + 1}")

def ingest_file(file_path):
    if not file_path.lower().endswith('.pdf'):
        print(f"Skipping non-PDF file: {file_path}")
        return
    
    print(f"Starting to ingest file: {file_path}")
    loader = PyPDFLoader(file_path)
    loaded_documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", " ", ""]
    )
    
    documents = []
    for original_doc in loaded_documents:
        chunks = text_splitter.split_documents([original_doc])
        for chunk_order, chunk in enumerate(chunks):
            chunk.metadata['chunk_order'] = chunk_order
        documents.extend(chunks)
    
    uuids = [str(uuid4()) for _ in range(len(documents))]
    print(f"Adding {len(documents)} documents to the vector store")

    texts = [doc.page_content for doc in documents]
    embeddings_data = model.encode(texts).tolist()

    book_id = os.path.basename(file_path)
    to_upsert = []
    for uuid, embedding, doc in zip(uuids, embeddings_data, documents):
        page_number = doc.metadata.get('page', 0) + 1 
        metadata = {
            'text': doc.page_content,
            'book_id': book_id,
            'page_number': page_number,
            'chunk_order': doc.metadata['chunk_order']
        }
        to_upsert.append((uuid, embedding, metadata))
    
    batch_upsert(index, to_upsert, batch_size=50)
    print(f"Finished ingesting file: {file_path}")

def query_page(book_id, page_number):
    """Query all text chunks from a specific book and page number."""
    dummy_vector = [0.0] * model.get_sentence_embedding_dimension()
    filter = {
        'book_id': {'$eq': book_id},
        'page_number': {'$eq': page_number}
    }
    
    try:
        results = index.query(
            vector=dummy_vector,
            top_k=10000,
            include_metadata=True,
            filter=filter
        )
        sorted_chunks = sorted(results.matches, key=lambda x: x.metadata['chunk_order'])
        return [chunk.metadata['text'] for chunk in sorted_chunks]
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

def main_loop():
    while True:
        for filename in os.listdir(data_folder):
            if not filename.startswith("_"):
                file_path = os.path.join(data_folder, filename)
                ingest_file(file_path)
                new_filename = "_" + filename
                new_file_path = os.path.join(data_folder, new_filename)
                os.rename(file_path, new_file_path)
        time.sleep(check_interval)

if __name__ == "__main__":
    main_loop()



# Example page number query usage
# pc = Pinecone(api_key=pinecone_api_key)
# index_name = "oxygen"
# index = pc.Index(index_name)
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# def query_page(book_id, page_number):
#     """Query all text chunks from a specific book and page number."""
#     dummy_vector = [0.0] * model.get_sentence_embedding_dimension()
#     filter = {
#         'book_id': {'$eq': book_id},
#         'page_number': {'$eq': page_number}
#     }
    
#     try:
#         results = index.query(
#             vector=dummy_vector,
#             top_k=10000,
#             include_metadata=True,
#             filter=filter
#         )
#         # Sort chunks by their order within the page
#         sorted_chunks = sorted(results.matches, key=lambda x: x.metadata['chunk_order'])
#         return [chunk.metadata['text'] for chunk in sorted_chunks]
#     except Exception as e:
#         print(f"Error querying Pinecone: {e}")
#         return []
# book_id = "C_DSA.pdf"  # Original filename before processing
# page_number = 205  # 1-based page number

# results = query_page(book_id, page_number)
# for i, text in enumerate(results):
#     print(f"Chunk {i+1}: {text}")


# def retrieve_relevant_docs(query, top_k=5, model_name="sentence-transformers/all-MiniLM-L6-v2", index=None):
#     """
#     Retrieve relevant documents from Pinecone based on the query.

#     Args:
#         query (str): The query string.
#         top_k (int): Number of top results to return.
#         model_name (str): Name of the Sentence Transformer model to use.
#         index: Pinecone index object.

#     Returns:
#         list: List of tuples containing (similarity, doc_id, chunk_id, chunk).
#     """
#     if index is None:
#         raise ValueError("Pinecone index must be provided.")

#     # Initialize the Sentence Transformer model
#     model = SentenceTransformer(model_name)

#     # Generate query embedding
#     query_embedding = model.encode(query).tolist()

#     # Query Pinecone index
#     try:
#         query_results = index.query(
#             vector=query_embedding,
#             top_k=top_k,
#             include_metadata=True
#         )
#     except Exception as e:
#         print(f"Error querying Pinecone: {e}")
#         return []

#     # Process results
#     results = []
#     for match in query_results.matches:
#         similarity = match.score  # Pinecone returns cosine similarity
#         doc_id = match.metadata.get("book_id", "unknown")  # Retrieve book_id from metadata
#         chunk_id = match.id  # Use the Pinecone ID as chunk_id
#         chunk = match.metadata.get("text", "")  # Retrieve text from metadata
#         results.append((similarity, doc_id, chunk_id, chunk))

#     return results


# # Initialize Pinecone
# pc = Pinecone(api_key=pinecone_api_key)
# index = pc.Index(index_name)

# # Query for relevant documents
# query = "code of for loop?"
# results = retrieve_relevant_docs(query, top_k=3, index=index)

# # Display results
# for similarity, doc_id, chunk_id, chunk in results:
#     print(f"Similarity: {similarity:.4f}")
#     print(f"Book ID: {doc_id}")
#     print(f"Chunk ID: {chunk_id}")
#     print(f"Text: {chunk}")
#     print("-" * 50)
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

class Retrieval:
    def __init__(self, db_path="../Database"):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="nutrition")
    
    def retrieve_documents(self, query, top_k=1):
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        
        retrieved_docs = []
        for doc_id, metadata in zip(results["ids"], results["metadatas"]):
            retrieved_docs.append({"id": doc_id, "metadata": metadata})
        
        return retrieved_docs

if __name__ == "__main__":
    rag = Retrieval()
    query = "ayam goreng"
    results = rag.retrieve_documents(query)
    for result in results:
        print(result)
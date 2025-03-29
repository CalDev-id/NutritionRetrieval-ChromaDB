import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Retrieval:
    def __init__(self, db_path="../Database"):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="nutrition")
    
    def retrieve_documents(self, query, top_k=5):
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        retrieved_docs = []

        for metadata_list in results["metadatas"]:  
            if metadata_list:  
                retrieved_docs.extend(metadata_list)

        return retrieved_docs

    
    def sentence_similarity(self, query, retrieved_docs):
        query_embedding = self.embedding_model.encode([query])
        resource_embeddings = self.embedding_model.encode([r['name'] for r in retrieved_docs])
        
        similarities = cosine_similarity(query_embedding, resource_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        
        return retrieved_docs[best_match_idx]

if __name__ == "__main__":
    rag = Retrieval()
    query = "ayam goreng"
    
    # Get top 5 relevant documents
    retrieved_docs = rag.retrieve_documents(query, top_k=5)
    
    # Find the most similar document
    best_match = rag.sentence_similarity(query, retrieved_docs)
    
    print("Best Match:", best_match)
    print("--"*50)
    print("perbandingan :", retrieved_docs)


# if __name__ == "__main__":
#     rag = Retrieval()
#     query = "ayam goreng"
#     results = rag.retrieve_documents(query)
#     for result in results:
#         print(result)
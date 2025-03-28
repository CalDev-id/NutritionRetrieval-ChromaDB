import json
import chromadb
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import pandas as pd

class Preprocessor:
    def __init__(self):
        csv_file = '../rawdata/nutrition.csv'  
        json_file = 'data_barang.json'
        db_path = "../Database"
        
        self.json_file = json_file  # Simpan agar bisa digunakan di method lain
        self.db_path = db_path
        
        self.convert_csv_to_json(csv_file, json_file)
        self.data = self.load_json()

        self.embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="nutrition")

    def convert_csv_to_json(self, csv_path, json_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File CSV '{csv_path}' tidak ditemukan.")
        
        df = pd.read_csv(csv_path)
        df.to_json(json_path, orient='records', indent=4)

    def load_json(self):
        if not os.path.exists(self.json_file):
            raise FileNotFoundError(f"File JSON '{self.json_file}' tidak ditemukan.")
        
        with open(self.json_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    
    def clean_metadata(self, item):
        """ Menghapus nilai None dari metadata """
        return {k: (v if v is not None else "") for k, v in item.items()}

    def create_embeddings_and_store(self):
        combined_texts = [
            f"{item.get('name', '')} {item.get('calories', '')} {item.get('proteins', '')} {item.get('fat', '')} {item.get('carbohydrate', '')}"
            for item in self.data
        ]
        
        embeddings = self.embedding_model.encode(combined_texts).tolist()
        
        for idx, (embedding, item) in enumerate(zip(embeddings, self.data)):
            cleaned_metadata = self.clean_metadata(item)
            self.collection.add(
                ids=[str(idx)], 
                embeddings=[embedding], 
                metadatas=[cleaned_metadata]
            )
        
        print("Data telah diproses dan disimpan di ChromaDB.")

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.create_embeddings_and_store()

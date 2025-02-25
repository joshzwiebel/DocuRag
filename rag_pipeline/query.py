import os
import faiss
import numpy as np
from dotenv import load_dotenv
import logging
import sys
import json
from openai import OpenAI
from requests import HTTPError
import truststore

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_extraction.gdrive_extraction import GoogleDriveClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.DEBUG = True
class FaissQuery:
    def __init__(self, faiss_index_path):
        load_dotenv()
        self.faiss_index_path = faiss_index_path
        self.index = self.load_faiss_index()
        service_account_file = os.getenv("SERVICE_ACCOUNT_FILE")
        scopes = os.getenv("SCOPES").split(',')
        self.client = GoogleDriveClient(service_account_file, scopes)

    def load_faiss_index(self):
        if os.path.exists(self.faiss_index_path):
            logger.info("Loading FAISS index from %s", self.faiss_index_path)
            return faiss.read_index(self.faiss_index_path)
        else:
            logger.error("FAISS index file not found at %s", self.faiss_index_path)
            raise FileNotFoundError(f"FAISS index file not found at {self.faiss_index_path}")

    def query(self, query_vector, k=1):
        logger.info("Querying FAISS index with k=%d", k)
        query_vector = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        return distances, indices

    def get_file_content(self, file_id, mime_type):
        try:
            if mime_type == 'application/pdf':
                return self.client.get_pdf_text_content(file_id)
            elif mime_type.startswith('application/vnd.google-apps'):
                return self.client.export_file(file_id, mime_type)
            else:
                return self.client.get_file_content(file_id)
        except HTTPError as error:
            logger.error("An error occurred while retrieving file content: %s", error)
            return None

    def text_to_vector(self, text):
        # Use OpenAI API to get the embeddings
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        vector = response.data[0].embedding
        return vector

if __name__ == "__main__":
    truststore.inject_into_ssl()
    faiss_index_path = './vector_store/faiss_index.index'

    # Get user input
    user_input = input("Enter your query: ")

    faiss_query = FaissQuery(faiss_index_path)

    # Convert user input to query vector
    query_vector = faiss_query.text_to_vector(user_input)

    distances, indices = faiss_query.query(query_vector, k=1)

    logger.info("Query results:")
    files = faiss_query.client.list_files()
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        logger.info("Result %d: Index %d, Distance %f", i, idx, dist)
        file = files[idx]
        file_content = faiss_query.get_file_content(file['id'], file['mimeType'])
        if file_content:
            logger.info("File Content: %s", file_content)
        else:
            logger.info("File Content: Not found")


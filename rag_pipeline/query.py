import os
import faiss
import numpy as np
from dotenv import load_dotenv
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_extraction.gdrive_extraction import GoogleDriveClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaissQuery:
    def __init__(self, faiss_index_path):
        self.faiss_index_path = faiss_index_path
        self.index = self.load_faiss_index()
        load_dotenv()
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

    def get_file_info(self, file_id):
        try:
            file = self.client.get_file_content(file_id)
            return file
        except Exception as e:
            logger.error("Error retrieving file info: %s", e)
            return None

    def text_to_vector(self, text):
        # Dummy implementation, replace with actual text to vector conversion
        return np.random.rand(128).astype('float32')

if __name__ == "__main__":
    faiss_index_path = './vector_store/faiss_index.index'
    
    # Get user input
    user_input = input("Enter your query: ")

    faiss_query = FaissQuery(faiss_index_path)
    
    # Convert user input to query vector
    query_vector = faiss_query.text_to_vector(user_input)
    
    distances, indices = faiss_query.query(query_vector, k=1)

    logger.info("Query results:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        logger.info("Result %d: Index %d, Distance %f", i, idx, dist)
        file_info = faiss_query.get_file_info(idx)
        if file_info:
            logger.info("File Info: Name: %s, ID: %s, MimeType: %s", file_info['name'], file_info['id'], file_info['mimeType'])
        else:
            logger.info("File Info: Not found")
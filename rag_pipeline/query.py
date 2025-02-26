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
load_dotenv()
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
        service_account_file = os.getenv("SERVICE_ACCOUNT_FILE")
        scopes = os.getenv("SCOPES").split(',')
        self.client = GoogleDriveClient(service_account_file, scopes)
        self.index = faiss.read_index(faiss_index_path)
        self.metadata_path = os.path.splitext(faiss_index_path)[0] + '_metadata.json'
        self.metadata = self.load_metadata()

    def load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def query(self, query_text, k=5):
        query_vector = self.text_to_vector(query_text)
        distances, indices = self.index.search(np.array([query_vector]).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if str(idx) in self.metadata:
                result = self.metadata[str(idx)]
                results.append(result)
        
        return distances[0], results

    def get_file_content(self, file_id, mime_type):
        """Get text content from various file types."""
        try:
            # Google Workspace files need special export handling
            if mime_type.startswith('application/vnd.google-apps'):
                return self.client.export_file(file_id, mime_type)
            
            # All other supported document types
            supported_types = {
                # PDF files
                'application/pdf': 'pdf',
                
                # Microsoft Office formats
                'application/msword': 'doc',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                'application/vnd.ms-excel': 'xls',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                'application/vnd.ms-powerpoint': 'ppt',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
                
                # Text formats
                'text/plain': 'txt',
                'text/csv': 'csv',
                'text/markdown': 'md',
                'application/json': 'json',
                'application/xml': 'xml',
                'text/html': 'html',
                'application/rtf': 'rtf'
            }
            
            if mime_type in supported_types:
                return self.client.extract_text(file_id, mime_type)
            else:
                logger.warning(f"Unsupported mime type: {mime_type}")
                return None

        except Exception as error:
            logger.error(f"Error retrieving file {file_id}: {error}")
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

    distances, results = faiss_query.query(user_input, k=5)

    logger.info("Query results:")
    files = faiss_query.client.list_files()
    for i, (dist, result) in enumerate(zip(distances, results)):
        logger.info("Result %d: Distance %f", i, dist)
        file_id = result['file_id']
        mime_type = result['mime_type']
        file_content = faiss_query.get_file_content(file_id, mime_type)
        if file_content:
            logger.info("File Content: %s", file_content)
        else:
            logger.info("File Content: Not found")


import os
import faiss
import numpy as np
import sys
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_extraction.gdrive_extraction import GoogleDriveClient
from dotenv import load_dotenv
from googleapiclient.errors import HttpError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, faiss_index_path):
        load_dotenv()
        service_account_file = os.getenv("SERVICE_ACCOUNT_FILE")
        scopes = os.getenv("SCOPES").split(',')
        self.client = GoogleDriveClient(service_account_file, scopes)
        self.faiss_index_path = faiss_index_path
        self.index = self.load_or_create_faiss_index()

    def load_or_create_faiss_index(self):
        if os.path.exists(self.faiss_index_path):
            logger.info("Loading existing FAISS index from %s", self.faiss_index_path)
            return faiss.read_index(self.faiss_index_path)
        else:
            logger.info("Creating new FAISS index")
            return faiss.IndexFlatL2(1536)  # OpenAI's text-embedding-ada-002 outputs 1536-dimensional vectors

    def preprocess_file(self, content):
        # Convert the content to a vector using OpenAI's API
        vector = self.text_to_vector(content)
        return vector

    def text_to_vector(self, text):
        # Use OpenAI API to get the embeddings
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        vector = response.data[0].embedding
        return vector

    def chunk_text(self, text, chunk_size=1024):
        # Split text into chunks of specified size
        print(len([text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]))
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def store_in_faiss(self, vector):
        self.index.add(np.array([vector]))

    def save_faiss_index(self):
        faiss.write_index(self.index, self.faiss_index_path)

    def handle_file(self, file_id, mime_type):
        try:
            if mime_type == 'application/pdf':
                content = self.client.get_pdf_text_content(file_id)
            elif mime_type.startswith('application/vnd.google-apps'):
                # Handle Google Docs Editors files by exporting them
                content = self.client.export_file(file_id, mime_type)
            else:
                # Handle other file types if necessary
                content = self.client.get_file_content(file_id)
            return content
        except HttpError as error:
            logger.error("An error occurred while handling file %s: %s", file_id, error)
            return None

    def run(self):
        files = self.client.list_files()
        for file in files:
            content = self.handle_file(file['id'], file['mimeType'])
            if content:
                chunks = self.chunk_text(content)
                for chunk in chunks:
                    vector = self.preprocess_file(chunk)
                    self.store_in_faiss(vector)
        self.save_faiss_index()

if __name__ == "__main__":
    faiss_index_path = './vector_store/faiss_index.index'
    preprocessor = Preprocessor(faiss_index_path)
    preprocessor.run()
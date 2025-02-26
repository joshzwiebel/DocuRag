import os
import logging
import sys
from dotenv import load_dotenv
from openai import OpenAI
import truststore
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import the FaissQuery class
from rag_pipeline.query import FaissQuery

class RAGAgent:
    """
    An AI agent that uses Retrieval Augmented Generation (RAG) to answer user questions
    by retrieving relevant documents from a FAISS index and using them as context for LLM responses.
    """
    
    def __init__(self, faiss_index_path, model="gpt-4o"):
        """
        Initialize the RAG Agent.
        
        Args:
            faiss_index_path (str): Path to the FAISS index file
            model (str): The OpenAI model to use for generating responses
        """
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        
        # Initialize the FAISS query object
        self.faiss_query = FaissQuery(faiss_index_path)
        
        # Inject truststore for SSL certificate handling
        truststore.inject_into_ssl()
    
    def _format_context(self, results, distances):
        """
        Format the retrieved documents into a context string for the LLM.
        
        Args:
            results (list): List of document metadata and content
            distances (list): List of similarity scores
            
        Returns:
            str: Formatted context string
        """
        context = "Here are the most relevant documents to help answer the question:\n\n"
        
        for i, (result, distance) in enumerate(zip(results, distances)):
            # Get file content if available
            file_id = result.get('file_id')
            mime_type = result.get('mime_type')
            
            if file_id and mime_type:
                file_content = self.faiss_query.get_file_content(file_id, mime_type)
                if file_content:
                    # Add document metadata and content to context
                    context += f"Document {i+1} (Relevance: {1.0 - distance:.2f}):\n"
                    context += f"Title: {result.get('title', 'Unknown')}\n"
                    context += f"Type: {mime_type}\n"
                    context += f"Content: {file_content[:1000]}...\n\n"  # Truncate long content
        
        return context
    
    def answer_question(self, question, k=5, temperature=0.7):
        """
        Answer a user question using RAG.
        
        Args:
            question (str): The user's question
            k (int): Number of documents to retrieve
            temperature (float): Temperature for response generation
            
        Returns:
            str: The agent's response
        """
        try:
            # Retrieve relevant documents
            distances, results = self.faiss_query.query(question, k=k)
            
            # Format context from retrieved documents
            context = self._format_context(results, distances)
            
            # Create prompt for the LLM
            prompt = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. "
                                             "If the context doesn't contain relevant information to answer the question, "
                                             "acknowledge that and provide a general response based on your knowledge. "
                                             "Always cite your sources when using information from the context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nPlease provide a comprehensive answer based on the context provided."}
            ]
            
            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=temperature,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"I encountered an error while trying to answer your question: {str(e)}"
    
    def interactive_session(self):
        """
        Run an interactive session where the user can ask multiple questions.
        """
        print("RAG Agent initialized. Ask a question or type 'exit' to quit.")
        
        while True:
            question = input("\nYour question: ")
            
            if question.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break
                
            print("\nThinking...")
            answer = self.answer_question(question)
            print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    # Path to the FAISS index
    faiss_index_path = './vector_store/faiss_index.index'
    
    # Create and run the agent
    agent = RAGAgent(faiss_index_path)
    agent.interactive_session()

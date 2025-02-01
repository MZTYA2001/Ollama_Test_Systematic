from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time
import sys
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import chromadb
from chromadb.config import Settings

# Create necessary directories
os.makedirs('files', exist_ok=True)

class ChromaDBHandler:
    """Handler for ChromaDB operations with better error management."""
    
    def __init__(self):
        self._client = None
        self._settings = None
        self.init_settings()
    
    def init_settings(self):
        """Initialize ChromaDB settings."""
        try:
            self._settings = Settings(
                is_persistent=False,  # Use in-memory storage
                anonymized_telemetry=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB settings: {str(e)}")
            raise

    @property
    def client(self):
        """Get ChromaDB client with lazy initialization."""
        if self._client is None:
            try:
                self._client = chromadb.Client(settings=self._settings)
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
                raise
        return self._client

def init_chroma() -> Optional[Chroma]:
    """Initialize ChromaDB with proper settings and error handling."""
    try:
        handler = ChromaDBHandler()
        
        embeddings = OllamaEmbeddings(
            base_url='http://localhost:11434',
            model="mistral"
        )
        
        return Chroma(
            embedding_function=embeddings,
            client_settings=handler._settings,
            client=handler.client
        )
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {str(e)}")
        logger.error(f"ChromaDB initialization error: {str(e)}")
        return None

def init_session_state():
    """Initialize Streamlit session state with necessary components."""
    if 'template' not in st.session_state:
        st.session_state.template = """You are a knowledgeable chatbot, here to help with questions about the document. Your tone should be professional and informative.

        Context: {context}
        History: {history}

        User: {question}
        Chatbot:"""

    if 'prompt' not in st.session_state:
        st.session_state.prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=st.session_state.template,
        )

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            input_key="question"
        )

    if 'vectorstore' not in st.session_state:
        vectorstore = init_chroma()
        if vectorstore is None:
            st.error("Failed to initialize vector store. Please check the logs for details.")
            sys.exit(1)
        st.session_state.vectorstore = vectorstore

    if 'llm' not in st.session_state:
        try:
            st.session_state.llm = Ollama(
                base_url="http://localhost:11434",
                model="mistral",
                verbose=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
            )
        except Exception as e:
            st.error(f"Failed to initialize Ollama LLM: {str(e)}")
            logger.error(f"Ollama initialization error: {str(e)}")
            sys.exit(1)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def process_pdf(file) -> Optional[Chroma]:
    """Process uploaded PDF file with error handling."""
    try:
        file_path = os.path.join("files", f"{file.name}.pdf")
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(file.read())
        
        # Load and process PDF
        loader = PyPDFLoader(file_path)
        data = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(data)
        
        if not splits:
            st.warning("No text content found in the PDF.")
            return None
        
        # Create embeddings
        embeddings = OllamaEmbeddings(
            model="mistral",
            base_url="http://localhost:11434"
        )
        
        # Update vector store
        handler = ChromaDBHandler()
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            client_settings=handler._settings,
            client=handler.client
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        logger.error(f"PDF processing error: {str(e)}")
        return None

def init_qa_chain(retriever):
    """Initialize the QA chain with error handling."""
    try:
        return RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )
    except Exception as e:
        st.error(f"Failed to initialize QA chain: {str(e)}")
        logger.error(f"QA chain initialization error: {str(e)}")
        return None

def main():
    """Main application function."""
    try:
        st.title("PDF Chatbot")
        
        # Initialize session state
        init_session_state()
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["message"])
        
        # File uploader
        uploaded_file = st.file_uploader("Upload your PDF", type='pdf')
        
        if uploaded_file:
            file_path = os.path.join("files", f"{uploaded_file.name}.pdf")
            
            if not os.path.isfile(file_path):
                with st.status("Analyzing your document..."):
                    vectorstore = process_pdf(uploaded_file)
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.retriever = vectorstore.as_retriever()
            
            if 'retriever' in st.session_state and 'qa_chain' not in st.session_state:
                qa_chain = init_qa_chain(st.session_state.retriever)
                if qa_chain:
                    st.session_state.qa_chain = qa_chain
            
            # Chat interface
            if user_input := st.chat_input("You:", key="user_input"):
                # Add user message
                user_message = {"role": "user", "message": user_input}
                st.session_state.chat_history.append(user_message)
                
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Generate and display response
                with st.chat_message("assistant"):
                    with st.spinner("Assistant is typing..."):
                        try:
                            response = st.session_state.qa_chain(user_input)
                            
                            message_placeholder = st.empty()
                            full_response = ""
                            
                            # Simulate typing effect
                            for chunk in response['result'].split():
                                full_response += chunk + " "
                                time.sleep(0.05)
                                message_placeholder.markdown(full_response + "▌")
                            message_placeholder.markdown(full_response)
                            
                            # Add assistant message to history
                            chatbot_message = {"role": "assistant", "message": response['result']}
                            st.session_state.chat_history.append(chatbot_message)
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
                            logger.error(f"Response generation error: {str(e)}")
        else:
            st.write("Please upload a PDF file.")
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Main application error: {str(e)}")

if __name__ == "__main__":
    main()

import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
import requests
from typing import List, Dict, Any
import time
import random
import hashlib
import gc
import torch

# Document processing imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load environment variables
load_dotenv()

# Enhanced Professional CSS with better contrast and modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --primary-blue: #2563eb;
        --primary-purple: #7c3aed;
        --dark-blue: #1e40af;
        --light-blue: #eff6ff;
        --gray-50: #f8fafc;
        --gray-100: #f1f5f9;
        --gray-200: #e2e8f0;
        --gray-300: #cbd5e1;
        --gray-400: #94a3b8;
        --gray-500: #64748b;
        --gray-600: #475569;
        --gray-700: #334155;
        --gray-800: #1e293b;
        --gray-900: #0f172a;
        --success-green: #10b981;
        --warning-orange: #f59e0b;
        --error-red: #ef4444;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-success: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .main {
        font-family: 'Inter', sans-serif;
        background: var(--gray-50);
        color: var(--gray-800);
    }
    
    /* Header Styles */
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        color: var(--gray-900);
        text-align: center;
        margin: 3rem 0 1rem 0;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.04em;
        line-height: 1.1;
    }
    
    .sub-header {
        font-size: 1.25rem;
        color: var(--gray-600);
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
        letter-spacing: -0.01em;
    }
    
    /* Card Components */
    .card {
        background: #ffffff;
        border: 1px solid var(--gray-200);
        border-radius: 16px;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
        position: relative;
    }
    
    .card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
        border-color: var(--gray-300);
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-primary);
    }
    
    /* Answer Container */
    .answer-container {
        background: #ffffff;
        border: 1px solid var(--gray-200);
        border-radius: 16px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    
    .answer-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-primary);
    }
    
    .answer-text {
        color: var(--gray-800);
        font-size: 1.1rem;
        line-height: 1.75;
        margin: 0;
        font-weight: 400;
    }
    
    .answer-header {
        color: var(--gray-900);
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Document Cards */
    .document-card {
        background: #ffffff;
        border: 2px solid var(--gray-200);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .document-card:hover {
        border-color: var(--primary-blue);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    .document-card.active {
        border-color: var(--primary-blue);
        background: var(--light-blue);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    .document-card.active::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
    }
    
    .document-title {
        font-weight: 600;
        color: var(--gray-900);
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .document-meta {
        color: var(--gray-600);
        font-size: 0.875rem;
        display: flex;
        gap: 1rem;
        align-items: center;
    }
    
    .document-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .document-status.active {
        background: var(--light-blue);
        color: var(--primary-blue);
    }
    
    .document-status.inactive {
        background: var(--gray-100);
        color: var(--gray-600);
    }
    
    /* Source Container */
    .source-container {
        background: var(--gray-50);
        border: 1px solid var(--gray-200);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .source-container:hover {
        background: #ffffff;
        box-shadow: var(--shadow-sm);
        border-color: var(--gray-300);
        transform: translateY(-1px);
    }
    
    .source-header {
        color: var(--primary-blue);
        font-weight: 600;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .source-text {
        color: var(--gray-700);
        line-height: 1.6;
        font-size: 0.9rem;
    }
    
    /* Status Messages */
    .status-success {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        color: var(--gray-800);
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--success-green);
        margin: 1.5rem 0;
        font-weight: 500;
        box-shadow: var(--shadow-sm);
    }
    
    .status-info {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        color: var(--gray-800);
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary-blue);
        margin: 1.5rem 0;
        font-weight: 500;
        box-shadow: var(--shadow-sm);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        color: var(--gray-800);
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--warning-orange);
        margin: 1.5rem 0;
        font-weight: 500;
        box-shadow: var(--shadow-sm);
    }
    
    /* Upload Area */
    .upload-container {
        border: 2px dashed var(--gray-300);
        border-radius: 16px;
        padding: 4rem 2rem;
        text-align: center;
        background: #ffffff;
        margin: 2rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--gradient-primary);
        opacity: 0;
        transition: opacity 0.3s ease;
        z-index: -1;
    }
    
    .upload-container:hover {
        border-color: var(--primary-blue);
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .upload-container:hover::before {
        opacity: 0.03;
    }
    
    .upload-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--gray-900);
        margin-bottom: 0.5rem;
    }
    
    .upload-subtitle {
        color: var(--gray-600);
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 2rem;
        margin-top: 2rem;
    }
    
    .feature-item {
        text-align: center;
        padding: 1rem;
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    .feature-title {
        font-weight: 600;
        color: var(--gray-900);
        margin-bottom: 0.5rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .feature-description {
        color: var(--gray-600);
        font-size: 0.9rem;
    }
    
    /* Button Styles */
    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
    }
    
    /* Input Styles */
    .stTextInput > div > div > input {
        border: 2px solid var(--gray-300);
        border-radius: 8px;
        padding: 0.875rem 1rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: #ffffff;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-blue);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        outline: none;
    }
    
    /* Loading Animation */
    .loading-text {
        font-family: 'JetBrains Mono', monospace;
        color: var(--primary-blue);
        font-weight: 500;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--gray-900);
        margin: 2rem 0 1rem 0;
        position: relative;
        padding-left: 1rem;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: var(--gradient-primary);
        border-radius: 2px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        color: var(--gray-500);
        font-size: 0.9rem;
        border-top: 1px solid var(--gray-200);
        margin-top: 4rem;
        background: #ffffff;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header[data-testid="stHeader"] {display: none;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .answer-container {
            padding: 1.5rem;
        }
        .document-card {
            padding: 1rem;
        }
        .feature-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        .section-header {
            font-size: 1.5rem;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main {
            background: var(--gray-900);
            color: var(--gray-100);
        }
    }
</style>
""", unsafe_allow_html=True)

# Sarcastic loading messages
LOADING_MESSAGES = [
    "Teaching AI to read... again",
    "Convincing the model to pay attention",
    "Bribing the neural networks with coffee",
    "Deciphering your PDF's hieroglyphics", 
    "Asking the AI to put down TikTok and focus",
    "Waiting for the model to finish its existential crisis",
    "Converting PDF chaos into something useful",
    "Translating human logic for silicon brains",
    "Performing computational wizardry",
    "Making the impossible seem routine",
    "Channeling digital telepathy",
    "Wrestling with uncooperative algorithms",
    "Applying artificial intelligence to natural confusion",
    "Coaxing sense from digital mayhem",
    "Teaching machines to think like humans (good luck with that)",
    "Debugging reality, one document at a time"
]

def get_random_loading_message():
    return random.choice(LOADING_MESSAGES)

# --- Environment Variables ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    st.markdown("""
    <div class="status-warning">
        <strong>Configuration Required</strong><br>
        Please set the MISTRAL_API_KEY environment variable to use this application.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- Utility Functions ---
def get_document_hash(file_content):
    """Generate unique hash for document content."""
    return hashlib.md5(file_content).hexdigest()

def initialize_document_storage():
    """Initialize session state for document storage."""
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
    if 'active_doc_id' not in st.session_state:
        st.session_state.active_doc_id = None

def clear_gpu_memory():
    """Clear GPU memory to prevent tensor errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# --- Enhanced Document Processor ---
class DocumentProcessor:
    """Enhanced document processor with better error handling."""
    
    @staticmethod
    def create_embeddings():
        """Create embeddings model with proper error handling."""
        try:
            clear_gpu_memory()  # Clear memory before loading
            
            # Use a more stable embedding model
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # More stable model
                model_kwargs={
                    'device': 'cpu',  # Force CPU to avoid tensor issues
                    'trust_remote_code': False
                },
                encode_kwargs={
                    'normalize_embeddings': False,
                    'batch_size': 32
                }
            )
            return embeddings
        except Exception as e:
            st.error(f"Error loading embedding model: {str(e)}")
            # Fallback to even simpler model
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': False}
                )
                return embeddings
            except:
                return None
    
    @staticmethod
    def load_and_process_document(file_path: str, doc_id: str, filename: str):
        """Load and process PDF with enhanced error handling."""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Loading phase
            status_text.markdown(f'<div class="loading-text">Loading "{filename}"...</div>', unsafe_allow_html=True)
            progress_bar.progress(15)
            time.sleep(0.5)
            
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                st.error("❌ No content found in PDF. Please upload a valid PDF file.")
                return None
            
            progress_bar.progress(30)
            
            # Text splitting phase
            status_text.markdown(f'<div class="loading-text">{get_random_loading_message()}</div>', unsafe_allow_html=True)
            progress_bar.progress(45)
            time.sleep(0.8)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Smaller chunks for stability
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            progress_bar.progress(60)
            
            # Embedding phase
            status_text.markdown(f'<div class="loading-text">Creating search index (or teaching AI to find things)...</div>', unsafe_allow_html=True)
            progress_bar.progress(75)
            time.sleep(1.0)
            
            # Create embeddings with error handling
            embeddings = DocumentProcessor.create_embeddings()
            if not embeddings:
                st.error("❌ Failed to load embedding model. Please try again.")
                return None
            
            # Create vector store with smaller batches
            try:
                # Process in smaller batches to avoid memory issues
                batch_size = 50
                all_vectorstores = []
                
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    if i == 0:
                        vectorstore = Chroma.from_documents(batch_chunks, embeddings)
                    else:
                        batch_vectorstore = Chroma.from_documents(batch_chunks, embeddings)
                        # Merge vectorstores (simplified approach)
                        vectorstore.add_documents(batch_chunks)
                    
                    # Update progress
                    batch_progress = 75 + (i / len(chunks)) * 20
                    progress_bar.progress(min(int(batch_progress), 95))
                
            except Exception as e:
                st.error(f"❌ Error creating search index: {str(e)}")
                return None
            
            progress_bar.progress(100)
            time.sleep(0.3)
            status_text.empty()
            progress_bar.empty()
            
            # Store document in session state
            st.session_state.documents[doc_id] = {
                'vectorstore': vectorstore,
                'filename': filename,
                'pages': len(documents),
                'chunks': len(chunks),
                'timestamp': time.time()
            }
            
            st.markdown(f"""
            <div class="status-success">
                <strong>✅ Document Processing Complete!</strong><br>
                Successfully processed <strong>{len(documents)} pages</strong> from "{filename}" into <strong>{len(chunks)} searchable segments</strong>.
            </div>
            """, unsafe_allow_html=True)
            
            return vectorstore
            
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            # Clear any partial state
            if doc_id in st.session_state.documents:
                del st.session_state.documents[doc_id]
            return None
        finally:
            # Cleanup
            clear_gpu_memory()

class EnhancedMistralQA:
    """Enhanced Mistral AI integration."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_answer(self, question: str, context: str, question_type: str = "general") -> str:
        """Get enhanced answer from Mistral AI."""
        try:
            # Dynamic prompting based on question type
            if question_type == "resume_analysis":
                system_prompt = """You are an expert HR analyst and career advisor. Analyze the provided resume/CV comprehensively:

ANALYZE:
- Technical and soft skills with proficiency levels
- Work experience, career progression, and achievements
- Educational background, certifications, and training
- Industry experience and domain expertise
- Leadership and project management experience

PROVIDE:
- Suitable roles and career opportunities 
- Strengths and competitive advantages
- Areas for development and skill gaps
- Industry fit and market positioning
- Salary range expectations (if applicable)
- Interview preparation recommendations

Be specific, actionable, and professional."""

            elif question_type == "summary":
                system_prompt = """You are a professional document analyst. Create comprehensive, structured summaries:

INCLUDE:
- Document type, purpose, and main objectives
- Key findings, results, conclusions, and recommendations
- Important data points, statistics, and metrics
- Significant people, organizations, dates, and locations
- Critical decisions, actions, or next steps
- Overall context, implications, and significance

STRUCTURE:
- Use clear headings and bullet points where appropriate
- Highlight the most important information
- Maintain professional tone and accuracy"""

            else:
                system_prompt = """You are an intelligent document analyst. Provide accurate, detailed responses:

APPROACH:
- Extract specific information directly from the document
- Make logical inferences when information is implied but not explicit
- Provide context and explanations for complex topics
- Clearly distinguish between facts and inferences
- State clearly when information is not available in the document

RESPONSE STYLE:
- Use professional, clear language
- Structure responses logically
- Include relevant details and examples
- Maintain accuracy and objectivity"""

            prompt = f"""{system_prompt}

DOCUMENT CONTENT:
{context}

QUESTION: {question}

PROFESSIONAL RESPONSE:"""

            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "mistral-medium",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.6,
                    "top_p": 0.9
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                return f"Unable to process your question at the moment. Please try again later. (Error: {response.status_code})"
                
        except requests.exceptions.Timeout:
            return "Request timed out. Please try again with a shorter question or check your connection."
        except Exception as e:
            return f"An error occurred while processing your request: {str(e)}"

class EnhancedDocumentQA:
    """Enhanced document Q&A system."""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.mistral_qa = EnhancedMistralQA(MISTRAL_API_KEY)
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,  # Reduced for better performance
                "fetch_k": 10,
                "lambda_mult": 0.7
            }
        )
    
    def classify_question(self, question: str) -> str:
        """Classify question type for optimal processing."""
        question_lower = question.lower()
        
        resume_keywords = ["resume", "cv", "role", "job", "position", "career", "hire", "candidate", "qualification", "experience", "skills", "employment", "work"]
        summary_keywords = ["summary", "summarize", "overview", "main points", "key findings", "conclusion", "abstract"]
        
        if any(word in question_lower for word in resume_keywords):
            return "resume_analysis"
        elif any(word in question_lower for word in summary_keywords):
            return "summary"
        else:
            return "general"
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Generate enhanced answer with context."""
        try:
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                return {
                    "answer": "I could not find relevant information in the document to answer this question. Please try rephrasing your question or ensure the document contains the information you're looking for.",
                    "sources": [],
                    "question_type": "general"
                }
            
            question_type = self.classify_question(question)
            
            # Prepare context with better formatting
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                page_info = f"[Source {i+1} - Page {doc.metadata.get('page', 'Unknown')}]"
                context_parts.append(f"{page_info}\n{doc.page_content.strip()}")
            
            context = "\n\n" + "="*50 + "\n\n".join(context_parts)
            
            # Get answer from Mistral
            answer = self.mistral_qa.get_answer(question, context, question_type)
            
            return {
                "answer": answer,
                "sources": relevant_docs,
                "question_type": question_type
            }
            
        except Exception as e:
            return {
                "answer": f"An error occurred while processing your question: {str(e)}. Please try again or contact support if the problem persists.",
                "sources": [],
                "question_type": "error"
            }

# --- Main Application ---

# Initialize document storage
initialize_document_storage()

# Header
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 class="main-header">AI Document Analyzer</h1>
    <p class="sub-header">Professional document analysis powered by advanced artificial intelligence</p>
</div>
""", unsafe_allow_html=True)

# Document Management Section
st.markdown('<h2 class="section-header">Document Management</h2>', unsafe_allow_html=True)

# Show loaded documents with enhanced UI
if st.session_state.documents:
    st.markdown("#### Currently Loaded Documents")
    
    # Create document cards
    for doc_id, doc_info in st.session_state.documents.items():
        is_active = st.session_state.active_doc_id == doc_id
        active_class = "active" if is_active else "inactive"
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            card_html = f"""
            <div class="document-card {'active' if is_active else ''}" onclick="selectDocument('{doc_id}')">
                <div class="document-title">{doc_info['filename']}</div>
                <div class="document-meta">
                    <span>📄 {doc_info['pages']} pages</span>
                    <span>🔍 {doc_info['chunks']} segments</span>
                    <div class="document-status {active_class}">
                        {'🟢 Active' if is_active else '⚪ Available'}
                    </div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
        
        with col2:
            if st.button("Select" if not is_active else "Active", 
                        key=f"select_{doc_id}",
                        disabled=is_active,
                        use_container_width=True):
                st.session_state.active_doc_id = doc_id
                st.session_state.qa_system = EnhancedDocumentQA(doc_info['vectorstore'])
                st.rerun()

# File Upload Section
st.markdown("#### Upload New Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    help="Select a PDF document to analyze with AI",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    file_content = uploaded_file.getvalue()
    doc_id = get_document_hash(file_content)
    filename = uploaded_file.name
    
    # Check if document already exists
    if doc_id not in st.session_state.documents:
        # Process new document
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        with st.spinner("Processing document..."):
            vectorstore = DocumentProcessor.load_and_process_document(tmp_file_path, doc_id, filename)
        
        # Cleanup temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        
        if vectorstore:
            st.session_state.active_doc_id = doc_id
            st.session_state.qa_system = EnhancedDocumentQA(vectorstore)
            st.rerun()
    else:
        st.markdown(f"""
        <div class="status-info">
            📋 <strong>Document Already Loaded</strong><br>
            "{filename}" is already in your document library. It has been selected as the active document.
        </div>
        """, unsafe_allow_html=True)
        st.session_state.active_doc_id = doc_id
        st.session_state.qa_system = EnhancedDocumentQA(st.session_state.documents[doc_id]['vectorstore'])

# Analysis Interface
if st.session_state.active_doc_id and st.session_state.active_doc_id in st.session_state.documents:
    active_doc = st.session_state.documents[st.session_state.active_doc_id]
    
    st.markdown("---")
    st.markdown('<h2 class="section-header">Document Analysis</h2>', unsafe_allow_html=True)
    
    # Active document indicator
    st.markdown(f"""
    <div class="status-info">
        📊 <strong>Analyzing:</strong> {active_doc['filename']}<br>
        <small>{active_doc['pages']} pages • {active_doc['chunks']} searchable segments</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick action buttons with enhanced styling
    st.markdown("#### Quick Analysis Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Document Summary", use_container_width=True, type="primary"):
            st.session_state.auto_question = "Provide a comprehensive summary of this document including its main purpose, key points, important findings, and conclusions."
        
        if st.button("💼 Career Analysis", use_container_width=True):
            st.session_state.auto_question = "Analyze this document as a professional profile. What are the person's key qualifications, experience, and what career opportunities would be most suitable?"
    
    with col2:
        if st.button("👤 Key Personnel", use_container_width=True):
            st.session_state.auto_question = "Who are the important people mentioned in this document? Include their names, titles, organizations, and roles."
        
        if st.button("🎯 Main Findings", use_container_width=True):
            st.session_state.auto_question = "What are the most important findings, results, achievements, or conclusions presented in this document?"
    
    # Custom question input
    st.markdown("#### Custom Question")
    
    default_question = st.session_state.get("auto_question", "")
    if default_question:
        del st.session_state.auto_question
        
    user_question = st.text_input(
        "Ask anything about the document:",
        value=default_question,
        placeholder="What would you like to know about this document?",
        key="question_input"
    )
    
    # Process question with enhanced loading
    if user_question and hasattr(st.session_state, 'qa_system'):
        loading_msg = get_random_loading_message()
        
        # Create loading container
        loading_container = st.empty()
        loading_container.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <div class="loading-text">{loading_msg}</div>
            <div style="margin-top: 1rem; opacity: 0.7;">This may take 10-30 seconds...</div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            result = st.session_state.qa_system.answer_question(user_question)
            loading_container.empty()
            
            # Display enhanced answer
            question_type_labels = {
                "resume_analysis": "🎯 Career Analysis",
                "summary": "📋 Document Summary", 
                "general": "💡 Analysis Result"
            }
            
            answer_label = question_type_labels.get(result.get("question_type", "general"), "💡 Analysis Result")
            
            st.markdown(f"""
            <div class="answer-container">
                <div class="answer-header">{answer_label}</div>
                <div class="answer-text">{result['answer']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources with enhanced formatting
            if result["sources"]:
                with st.expander(f"📚 Source References ({len(result['sources'])} sections found)", expanded=False):
                    for i, doc in enumerate(result["sources"], 1):
                        page_num = doc.metadata.get('page', 'Unknown')
                        preview = doc.page_content[:600] + "..." if len(doc.page_content) > 600 else doc.page_content
                        
                        st.markdown(f"""
                        <div class="source-container">
                            <div class="source-header">📄 Reference {i} | Page {page_num}</div>
                            <div class="source-text">{preview}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        except Exception as e:
            loading_container.empty()
            st.error(f"An error occurred while processing your question: {str(e)}")

elif not st.session_state.documents:
    # Enhanced welcome screen
    st.markdown("""
    <div class="upload-container">
    <div class="upload-title">🚀 Ready for Intelligent Document Analysis</div>
    <div class="upload-subtitle">Upload PDF documents and unlock AI-powered insights</div>
    
    <div class="feature-grid">
        <div class="feature-item">
            <div class="feature-icon">📚</div>
            <div class="feature-title">Multi-Document Support</div>
            <div class="feature-description">Load multiple PDFs and switch between them seamlessly</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">Advanced AI Analysis</div>
            <div class="feature-description">Powered by state-of-the-art language models for accurate insights</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Smart Question Answering</div>
            <div class="feature-description">Ask complex questions and get detailed, contextual answers</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Professional Summaries</div>
            <div class="feature-description">Generate comprehensive summaries and key insights</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">💼</div>
            <div class="feature-title">Career Analysis</div>
            <div class="feature-description">Specialized analysis for resumes and professional documents</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Source Citations</div>
            <div class="feature-description">Every answer includes verifiable source references</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="status-info">
        📋 <strong>Documents Loaded</strong><br>
        Please select a document above to begin analysis, or upload a new document.
    </div>
    """, unsafe_allow_html=True)

# Document management controls
if st.session_state.documents:
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if st.button("🗑️ Clear All Documents", type="secondary"):
            st.session_state.documents = {}
            st.session_state.active_doc_id = None
            if hasattr(st.session_state, 'qa_system'):
                delattr(st.session_state, 'qa_system')
            clear_gpu_memory()  # Clear memory after removing documents
            st.rerun()
    
    with col2:
        st.metric("Documents Loaded", len(st.session_state.documents))
    
    with col3:
        if st.session_state.active_doc_id:
            st.success("✅ Ready")
        else:
            st.warning("⚠️ Select Document")

# Enhanced footer
st.markdown("""
<div class="footer">
    <strong>AI Document Analyzer</strong> | Professional Document Intelligence Platform<br>
    <small>Powered by Advanced Natural Language Processing • Built with Security & Privacy in Mind</small>
</div>
""", unsafe_allow_html=True)
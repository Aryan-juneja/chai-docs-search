import streamlit as st
import os
from typing import List, Dict, Any
import time
from datetime import datetime
import logging
import re
import json
# Langchain imports
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
# Pinecone
from pinecone import Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== API KEY CONFIGURATION =====
# Uncomment and set your API keys here for quick testing (NOT recommended for production)
# os.environ["PINECONE_API_KEY"] = "your-pinecone-api-key-here"
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"
# ===================================
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Chai Docs Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful dark theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');
    
    /* Root variables for dark theme */
    :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #1a1a1a;
        --bg-tertiary: #2a2a2a;
        --bg-card: #1e1e1e;
        --bg-code: #0d1117;
        --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --text-primary: #ffffff;
        --text-secondary: #b0b0b0;
        --text-muted: #808080;
        --text-code: #f8f8f2;
        --accent-primary: #667eea;
        --accent-secondary: #764ba2;
        --accent-success: #10b981;
        --accent-warning: #f59e0b;
        --accent-error: #ef4444;
        --accent-code: #79c0ff;
        --border-primary: #3a3a3a;
        --border-secondary: #2a2a2a;
        --border-code: #30363d;
        --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.5);
        --glow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Main app styling */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main header with animated gradient */
    .main-header {
        background: var(--bg-gradient);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        transform: translateX(-100%);
        animation: shine 3s infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .main-header h1 {
        color: var(--text-primary) !important;
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9) !important;
        margin: 1rem 0 0 0;
        font-size: 1.2rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Search container with glassmorphism effect */
    .search-container {
        background: rgba(30, 30, 30, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-primary);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    
    .search-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
    }
    
    .search-container h3 {
        color: var(--text-primary) !important;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Enhanced code block styling */
    .code-block {
        background: var(--bg-code) !important;
        border: 1px solid var(--border-code) !important;
        border-radius: 12px !important;
        margin: 1rem 0 !important;
        padding: 1.5rem !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 0.9rem !important;
        line-height: 1.6 !important;
        overflow-x: auto !important;
        position: relative !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .code-block::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
        border-radius: 12px 12px 0 0;
    }
    
    .code-block code {
        color: var(--text-code) !important;
        background: transparent !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 0.9rem !important;
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
    
    /* Inline code styling */
    .inline-code {
        background: rgba(121, 192, 255, 0.1) !important;
        color: var(--accent-code) !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 0.85rem !important;
        border: 1px solid rgba(121, 192, 255, 0.2) !important;
    }
    
    /* Enhanced link styling */
    .response-link {
        color: var(--accent-primary) !important;
        text-decoration: none !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        background: rgba(102, 126, 234, 0.1) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.3rem !important;
        margin: 0.1rem !important;
    }
    
    .response-link::after {
        content: '🔗' !important;
        font-size: 0.8rem !important;
        opacity: 0.7 !important;
    }
    
    .response-link:hover {
        color: var(--text-primary) !important;
        background: var(--accent-primary) !important;
        box-shadow: var(--glow) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Response container with better formatting */
    .response-container {
        background: var(--bg-card);
        border: 1px solid var(--border-primary);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-sm);
        border-left: 4px solid var(--accent-primary);
        color: var(--text-primary) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        line-height: 1.8;
        font-size: 1rem;
    }
    
    .response-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.05), transparent);
        transition: left 0.5s;
    }
    
    .response-container:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-color: var(--accent-primary);
        background: rgba(30, 30, 30, 0.9);
    }
    
    .response-container:hover::before {
        left: 100%;
    }
    
    .response-container h1,
    .response-container h2,
    .response-container h3,
    .response-container h4,
    .response-container h5,
    .response-container h6 {
        color: var(--text-primary) !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        font-weight: 600 !important;
    }
    
    .response-container p {
        color: var(--text-secondary) !important;
        margin-bottom: 1rem !important;
        line-height: 1.8 !important;
    }
    
    .response-container ul,
    .response-container ol {
        color: var(--text-secondary) !important;
        margin-left: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    .response-container li {
        margin-bottom: 0.5rem !important;
        line-height: 1.6 !important;
    }
    
    .response-container strong {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    /* Result cards with hover effects */
    .result-card {
        background: var(--bg-card);
        border: 1px solid var(--border-primary);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-sm);
        border-left: 4px solid var(--accent-primary);
        color: var(--text-primary) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.05), transparent);
        transition: left 0.5s;
    }
    
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: var(--accent-primary);
        background: rgba(30, 30, 30, 0.9);
    }
    
    .result-card:hover::before {
        left: 100%;
    }
    
    .result-card h4 {
        color: var(--text-primary) !important;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .result-card p {
        color: var(--text-secondary) !important;
        line-height: 1.7;
        margin-bottom: 1rem;
    }
    
    .result-card strong {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    /* Category badges with vibrant colors */
    .category-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-primary) !important;
        background: var(--bg-gradient);
        border-radius: 25px;
        margin-right: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: var(--shadow-sm);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Links with hover effects */
    .url-link {
        color: var(--accent-primary) !important;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .url-link::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 0;
        height: 2px;
        background: var(--accent-primary);
        transition: width 0.3s ease;
    }
    
    .url-link:hover {
        color: var(--accent-secondary) !important;
        text-shadow: 0 0 8px rgba(102, 126, 234, 0.5);
    }
    
    .url-link:hover::after {
        width: 100%;
    }
    
    /* Stats container with neon effect */
    .stats-container {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid var(--accent-success);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        color: var(--accent-success) !important;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 20px rgba(16, 185, 129, 0.2); }
        to { box-shadow: 0 0 30px rgba(16, 185, 129, 0.4); }
    }
    
    .stats-container strong {
        color: var(--text-primary) !important;
    }
    
    /* Sidebar styling */
    .sidebar-info {
        background: var(--bg-card);
        border: 1px solid var(--border-primary);
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-sm);
    }
    
    .sidebar-info h3 {
        color: var(--text-primary) !important;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    .sidebar-info p {
        color: var(--text-secondary) !important;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Streamlit component styling */
    .stTextInput > div > div > input {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border-primary) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: var(--glow) !important;
        background-color: var(--bg-tertiary) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* Button styling with animation */
    .stButton > button {
        background: var(--bg-gradient) !important;
        color: var(--text-primary) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-lg) !important;
        filter: brightness(1.1) !important;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Expander styling */
    .stExpander {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 12px !important;
        margin-bottom: 1rem !important;
        overflow: hidden !important;
    }
    
    .stExpander > div > div {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > div {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 8px !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        color: var(--text-primary) !important;
    }
    
    .stSlider .stSlider > div > div > div > div > div {
        background-color: var(--accent-primary) !important;
    }
    
    /* Sidebar background */
    .css-1d391kg, .css-1y4p8pa {
        background-color: var(--bg-secondary) !important;
    }
    
    .css-1d391kg .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Regular text */
    p, div, span, .stMarkdown {
        color: var(--text-secondary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: var(--bg-tertiary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 8px !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid var(--accent-success) !important;
        color: var(--accent-success) !important;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid var(--accent-error) !important;
        color: var(--accent-error) !important;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid var(--accent-warning) !important;
        color: var(--accent-warning) !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: var(--accent-primary) !important;
    }
    
    /* Footer with gradient text */
    .footer-text {
        text-align: center;
        background: var(--bg-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 3rem;
        padding: 2rem;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-secondary);
    }
    
    /* Form styling */
    .stForm {
        background: transparent !important;
        border: none !important;
    }
    
    /* Metric styling */
    .metric-container {
        background: var(--bg-card);
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    /* Loading animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .search-container {
            padding: 1.5rem;
        }
        
        .result-card {
            padding: 1.5rem;
        }
        
        .response-container {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class ChaiDocsSearchApp:
    def __init__(self):
        self.setup_session_state()
        self.initialize_components()
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'llm' not in st.session_state:
            st.session_state.llm = None
    
    def initialize_components(self):
        """Initialize Pinecone, embeddings, and LLM"""
        try:
            # Check for API keys with better error handling
            pinecone_key = os.getenv("PINECONE_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            
            # Try to get from Streamlit secrets if environment variables not found
            if not pinecone_key or not openai_key:
                try:
                    if not pinecone_key:
                        pinecone_key = st.secrets.get("PINECONE_API_KEY")
                    if not openai_key:
                        openai_key = st.secrets.get("OPENAI_API_KEY")
                except Exception:
                    # Secrets not available, continue with None values
                    pass
            
            if not pinecone_key or not openai_key:
                st.error("⚠️ API Keys Missing!")
                st.markdown("""
                Please set your API keys using one of these methods:
                
                **Method 1: Environment Variables (Recommended)**
                ```bash
                export PINECONE_API_KEY="your-pinecone-api-key"
                export OPENAI_API_KEY="your-openai-api-key"
                ```
                
                **Method 2: Streamlit Secrets**
                Create a file `.streamlit/secrets.toml` in your project directory:
                ```toml
                PINECONE_API_KEY = "your-pinecone-api-key"
                OPENAI_API_KEY = "your-openai-api-key"
                ```
                
                **Method 3: Set in Code (Not Recommended for Production)**
                Add these lines before running the app:
                ```python
                import os
                os.environ["PINECONE_API_KEY"] = "your-pinecone-api-key"
                os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
                ```
                """)
                st.stop()
            
            # Initialize components
            if st.session_state.vector_store is None:
                with st.spinner("🔄 Initializing search engine..."):
                    # Initialize Pinecone
                    pc = Pinecone(api_key=pinecone_key)
                    index_name = "chai-docs-index"
                    
                    # Check if index exists
                    existing_indexes = [idx.name for idx in pc.list_indexes()]
                    if index_name not in existing_indexes:
                        st.error(f"❌ Pinecone index '{index_name}' not found. Please run the data loading script first.")
                        st.stop()
                    
                    # Initialize embeddings and vector store
                    embeddings = OpenAIEmbeddings(
                        model="text-embedding-3-large",
                        openai_api_key=openai_key
                    )
                    
                    index = pc.Index(index_name)
                    st.session_state.vector_store = PineconeVectorStore(
                        index=index,
                        embedding=embeddings
                    )
                    
                    # Initialize LLM
                    st.session_state.llm = ChatOpenAI(
                        model="gpt-4.1-mini",
                        temperature=0.1,
                        openai_api_key=openai_key
                    )
                    
                    st.success("✅ Search engine initialized successfully!")
        
        except Exception as e:
            st.error(f"❌ Error initializing components: {str(e)}")
            logger.error(f"Initialization error: {str(e)}")
            st.stop()
    
    def create_search_prompt(self, query: str, context_docs: List[Document]) -> str:
        """Create a comprehensive prompt for the LLM with JSON response format"""
        context_text = ""
        for i, doc in enumerate(context_docs, 1):
            context_text += f"""
Document {i}:
Title: {doc.metadata.get('title', 'N/A')}
Category: {doc.metadata.get('category', 'N/A')}
Topic: {doc.metadata.get('topic', 'N/A')}
URL: {doc.metadata.get('source', 'N/A')}
Content: {doc.page_content}
---
"""
        
        prompt = f"""
You are a helpful assistant that provides detailed answers about programming and development topics based on the Chai Docs documentation.

Based on the following context documents, please answer the user's question and format your response as a JSON object with the following structure:

{{
    "summary": "Brief overview of the answer in 1-2 sentences",
    "detailed_answer": "Comprehensive explanation of the topic",
    "code_examples": [
        {{
            "language": "html/python/javascript/sql/bash",
            "description": "What this code does",
            "code": "actual code here"
        }}
    ],
    "key_points": [
        "Important point 1",
        "Important point 2",
        "Important point 3"
    ],
    "related_links": [
        {{
            "title": "Link title from source",
            "url": "actual URL from metadata",
            "description": "Brief description of what this link covers"
        }}
    ],
    "categories": ["category1", "category2"],
    "difficulty_level": "beginner/intermediate/advanced",
    "additional_resources": [
        "Suggestion 1 for further learning",
        "Suggestion 2 for further learning"
    ]
}}

CONTEXT DOCUMENTS:
{context_text}

USER QUESTION: {query}

Important guidelines:
1. Extract actual URLs from the document metadata for the related_links section
2. Include practical code examples when relevant
3. Make the detailed_answer comprehensive but well-structured
4. Ensure all JSON is properly formatted and valid
5. Use the categories from the source documents
6. Provide actionable key points

Return ONLY the JSON response, no additional text before or after.
"""
        return prompt
    def parse_json_response(self, response_text: str) -> Dict[Any, Any]:
        """Parse JSON response from LLM with error handling"""
        try:
            # Try to extract JSON from response if it contains extra text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Fallback to plain text if JSON parsing fails
            return {
                "summary": "Response formatting error",
                "detailed_answer": response_text,
                "code_examples": [],
                "key_points": [],
                "related_links": [],
                "categories": [],
                "difficulty_level": "unknown",
                "additional_resources": []
            }
    def search_and_generate_response(self, query: str, num_results: int = 5):
        """Search Pinecone and generate LLM response"""
        try:
            with st.spinner("🔍 Searching knowledge base..."):
                # Search for relevant documents
                search_results = st.session_state.vector_store.similarity_search(
                    query, 
                    k=num_results
                )
                
                if not search_results:
                    return None, [], "No relevant documents found for your query."
            
            with st.spinner("🤖 Generating comprehensive answer..."):
                # Create prompt and get LLM response
                prompt = self.create_search_prompt(query, search_results)
                
                response = st.session_state.llm.invoke([
                    SystemMessage(content="You are a knowledgeable programming instructor helping students learn from the Chai Docs."),
                    HumanMessage(content=prompt)
                ])
                
                return response.content, search_results, None
        
        except Exception as e:
            error_msg = f"Error during search: {str(e)}"
            logger.error(error_msg)
            return None, [], error_msg
    
    def display_search_results(self, results: List[Document]):
        """Display search results in an organized format"""
        if not results:
            return
            
        st.markdown("### 📚 Source Documents")
        
        for i, doc in enumerate(results, 1):
            with st.container():
                st.markdown(f"""
                <div class="result-card fade-in">
                    <h4>📄 Document {i}</h4>
                    <div style="margin-bottom: 15px;">
                        <span class="category-badge">{doc.metadata.get('category', 'Unknown').upper()}</span>
                        <strong>Topic:</strong> {doc.metadata.get('topic', 'N/A')}
                    </div>
                    <p><strong>Title:</strong> {doc.metadata.get('title', 'No title available')}</p>
                    <p><strong>Content Preview:</strong></p>
                    <div style="background: rgba(42, 42, 42, 0.5); padding: 15px; border-radius: 8px; font-size: 14px; line-height: 1.6; border-left: 3px solid var(--accent-primary);">
                        {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}
                    </div>
                    <p style="margin-top: 15px;"><strong>🔗 Source:</strong> 
                        <a href="{doc.metadata.get('source', '#')}" target="_blank" class="url-link">
                            {doc.metadata.get('source', 'No URL available')}
                        </a>
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    def add_to_history(self, query: str, response: str):
        """Add search to history"""
        history_item = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'query': query,
            'response': response[:200] + "..." if len(response) > 200 else response
        }
        st.session_state.search_history.insert(0, history_item)
        # Keep only last 10 searches
        st.session_state.search_history = st.session_state.search_history[:10]
    
    def display_sidebar(self):
        """Display sidebar with additional information and settings"""
        with st.sidebar:
            st.markdown("""
            <div class="sidebar-info">
                <h3>🚀 Chai Docs Search</h3>
                <p>Search through comprehensive programming tutorials and documentation with AI-powered insights.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Search settings
            st.markdown("### ⚙️ Search Settings")
            num_results = st.slider("Number of source documents", 3, 10, 5)
            
            # Available categories with improved styling
            st.markdown("### 📂 Available Categories")
            categories = ["HTML", "Git", "C Programming", "Django", "SQL", "DevOps"]
            for category in categories:
                st.markdown(f"""
                <div class="metric-container" style="margin-bottom: 0.5rem; padding: 0.5rem;">
                    <span style="color: var(--accent-primary);">•</span> {category}
                </div>
                """, unsafe_allow_html=True)
            
            # Search history
            if st.session_state.search_history:
                st.markdown("### 📋 Recent Searches")
                for item in st.session_state.search_history[:5]:
                    with st.expander(f"🕒 {item['timestamp']}", expanded=False):
                        st.markdown(f"**Query:** {item['query']}")
                        st.markdown(f"**Response:** {item['response']}")
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.search_history = []
                st.rerun()
            
            return num_results
    def display_json_response(self, json_response: Dict[Any, Any]):
        """Display formatted JSON response with enhanced styling"""
        
        # Summary section
        if json_response.get("summary"):
            st.markdown(f"""
            <div class="response-container" style="border-left: 4px solid var(--accent-success);">
                <h3 style="color: var(--accent-success); margin-bottom: 1rem;">📋 Quick Summary</h3>
                <p style="font-size: 1.1rem; font-weight: 500;">{json_response["summary"]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed answer section
        if json_response.get("detailed_answer"):
            st.markdown(f"""
            <div class="response-container">
                <h3 style="color: var(--accent-primary); margin-bottom: 1rem;">📚 Detailed Explanation</h3>
                <div style="line-height: 1.8;">
                    {self.format_detailed_answer(json_response["detailed_answer"])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Code examples section
        if json_response.get("code_examples"):
            st.markdown("""
            <div class="response-container" style="border-left: 4px solid var(--accent-code);">
                <h3 style="color: var(--accent-code); margin-bottom: 1rem;">💻 Code Examples</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i, example in enumerate(json_response["code_examples"], 1):
                st.markdown(f"""
                <div class="code-block">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h4 style="margin: 0; color: var(--accent-code);">Example {i}: {example.get('description', 'Code Example')}</h4>
                        <span style="background: rgba(121, 192, 255, 0.2); color: var(--accent-code); padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">
                            {example.get('language', 'code')}
                        </span>
                    </div>
                    <pre><code>{example.get('code', '')}</code></pre>
                </div>
                """, unsafe_allow_html=True)
        
        # Key points section
        if json_response.get("key_points"):
            st.markdown("""
            <div class="response-container" style="border-left: 4px solid var(--accent-warning);">
                <h3 style="color: var(--accent-warning); margin-bottom: 1rem;">🎯 Key Points</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for point in json_response["key_points"]:
                st.markdown(f"""
                <div style="display: flex; align-items: flex-start; margin-bottom: 0.8rem; padding: 0.8rem; background: rgba(245, 158, 11, 0.1); border-radius: 8px; border-left: 3px solid var(--accent-warning);">
                    <span style="color: var(--accent-warning); margin-right: 0.8rem; font-size: 1.2rem;">•</span>
                    <p style="margin: 0; color: var(--text-secondary); line-height: 1.6;">{point}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Related links section
        if json_response.get("related_links"):
            st.markdown("""
            <div class="response-container" style="border-left: 4px solid var(--accent-secondary);">
                <h3 style="color: var(--accent-secondary); margin-bottom: 1rem;">🔗 Related Resources</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for link in json_response["related_links"]:
                st.markdown(f"""
                <div style="margin-bottom: 1rem; padding: 1rem; background: rgba(118, 75, 162, 0.1); border-radius: 8px; border-left: 3px solid var(--accent-secondary);">
                    <h4 style="margin: 0 0 0.5rem 0; color: var(--text-primary);">
                        <a href="{link.get('url', '#')}" target="_blank" class="response-link" style="text-decoration: none; color: var(--accent-secondary);">
                            {link.get('title', 'Resource')}
                        </a>
                    </h4>
                    <p style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">{link.get('description', '')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Metadata section
        col1, col2 = st.columns(2)
        
        with col1:
            if json_response.get("categories"):
                st.markdown("""
                <div class="response-container" style="border-left: 4px solid var(--accent-primary);">
                    <h4 style="color: var(--accent-primary); margin-bottom: 1rem;">📂 Categories</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for category in json_response["categories"]:
                    st.markdown(f'<span class="category-badge">{category}</span>', unsafe_allow_html=True)
        
        with col2:
            if json_response.get("difficulty_level"):
                difficulty_colors = {
                    "beginner": "var(--accent-success)",
                    "intermediate": "var(--accent-warning)", 
                    "advanced": "var(--accent-error)"
                }
                color = difficulty_colors.get(json_response["difficulty_level"].lower(), "var(--accent-primary)")
                
                st.markdown(f"""
                <div class="response-container" style="border-left: 4px solid {color};">
                    <h4 style="color: {color}; margin-bottom: 1rem;">📊 Difficulty Level</h4>
                    <span style="background: {color}; color: var(--text-primary); padding: 0.5rem 1rem; border-radius: 20px; font-weight: 600; text-transform: capitalize;">
                        {json_response["difficulty_level"]}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        
        # Additional resources section
        if json_response.get("additional_resources"):
            st.markdown("""
            <div class="response-container" style="border-left: 4px solid var(--accent-success);">
                <h3 style="color: var(--accent-success); margin-bottom: 1rem;">🚀 Next Steps</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for resource in json_response["additional_resources"]:
                st.markdown(f"""
                <div style="display: flex; align-items: flex-start; margin-bottom: 0.8rem; padding: 0.8rem; background: rgba(16, 185, 129, 0.1); border-radius: 8px; border-left: 3px solid var(--accent-success);">
                    <span style="color: var(--accent-success); margin-right: 0.8rem; font-size: 1.2rem;">→</span>
                    <p style="margin: 0; color: var(--text-secondary); line-height: 1.6;">{resource}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def format_detailed_answer(self, text: str) -> str:
        """Format detailed answer text with proper HTML styling"""
        # Convert newlines to HTML breaks
        text = text.replace('\n', '<br>')
        
        # Format inline code (text between backticks)
        text = re.sub(r'`([^`]+)`', r'<span class="inline-code">\1</span>', text)
        
        # Format bold text
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        
        # Format italic text
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        
        return text
    
    def search_and_generate_response(self, query: str, num_results: int = 5):
        """Search Pinecone and generate JSON formatted LLM response"""
        try:
            with st.spinner("🔍 Searching knowledge base..."):
                # Search for relevant documents
                search_results = st.session_state.vector_store.similarity_search(
                    query, 
                    k=num_results
                )
                
                if not search_results:
                    return None, [], "No relevant documents found for your query."
            
            with st.spinner("🤖 Generating comprehensive answer..."):
                # Create prompt and get LLM response
                prompt = self.create_search_prompt(query, search_results)
                
                response = st.session_state.llm.invoke([
                    SystemMessage(content="You are a knowledgeable programming instructor helping students learn from the Chai Docs. Always respond with valid JSON format only."),
                    HumanMessage(content=prompt)
                ])
                
                # Parse JSON response
                json_response = self.parse_json_response(response.content)
                
                return json_response, search_results, None
        
        except Exception as e:
            error_msg = f"Error during search: {str(e)}"
            logger.error(error_msg)
            return None, [], error_msg
    def run(self):
        """Main application function"""
        # Header with enhanced styling
        st.markdown("""
        <div class="main-header">
            <h1>🔍 Chai Docs Intelligent Search</h1>
            <p>Ask questions about programming concepts and get comprehensive answers with source references</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        num_results = self.display_sidebar()
        
        # Main search interface
        st.markdown("""
        <div class="search-container">
            <h3>💬 Ask Your Programming Question</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a form for Enter key functionality
        with st.form(key='search_form', clear_on_submit=False):
            # Search input
            query = st.text_input(
                "Enter your question:",
                placeholder="e.g., How to create HTML forms? What are Git branches? Django model relationships?",
                help="Ask detailed questions about programming concepts covered in Chai Docs",
                key="search_input"
            )
            
            # Search button (will also trigger on Enter)
            search_clicked = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)
        
        # Example questions with enhanced styling
        with st.expander("💡 Example Questions"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="metric-container" style="margin-bottom: 1rem;">
                    <strong style="color: var(--accent-primary);">HTML & Web Development:</strong><br>
                    • How to create responsive HTML forms?<br>
                    • What are HTML semantic tags?<br>
                    • How to use Emmet shortcuts?
                </div>
                
                <div class="metric-container" style="margin-bottom: 1rem;">
                    <strong style="color: var(--accent-secondary);">Git & Version Control:</strong><br>
                    • How to create and merge Git branches?<br>
                    • What is Git stashing?<br>
                    • How to manage Git history?
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-container" style="margin-bottom: 1rem;">
                    <strong style="color: var(--accent-success);">Django & Python:</strong><br>
                    • How to create Django models?<br>
                    • What are Django template relationships?<br>
                    • How to use Jinja templates?
                </div>
                
                <div class="metric-container" style="margin-bottom: 1rem;">
                    <strong style="color: var(--accent-warning);">DevOps & Deployment:</strong><br>
                    • How to setup Nginx server?<br>
                    • What is Docker PostgreSQL setup?<br>
                    • How to configure SSL certificates?
                </div>
                """, unsafe_allow_html=True)
        
        # Search button and processing
        if search_clicked and query.strip():
            # Perform search
            json_response, results, error = self.search_and_generate_response(query, num_results)
            
            if error:
                st.error(f"❌ {error}")
            elif json_response and results:
                # Add to history (store summary for history)
                summary_for_history = json_response.get("summary", str(json_response)[:200])
                self.add_to_history(query, summary_for_history)
                
                # Display JSON formatted answer
                st.markdown("### 🎯 Comprehensive Answer")
                self.display_json_response(json_response)
                
                # Display source documents
                self.display_search_results(results)
                
                # Enhanced stats with visual metrics
                categories = list(set(doc.metadata.get('category', 'Unknown') for doc in results))
                st.markdown("### 📊 Search Statistics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3 style="color: var(--accent-primary); margin: 0; font-size: 2rem;">{len(results)}</h3>
                        <p style="margin: 0; color: var(--text-secondary);">Documents Found</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3 style="color: var(--accent-secondary); margin: 0; font-size: 2rem;">{len(categories)}</h3>
                        <p style="margin: 0; color: var(--text-secondary);">Categories</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3 style="color: var(--accent-success); margin: 0; font-size: 2rem;">AI</h3>
                        <p style="margin: 0; color: var(--text-secondary);">Powered</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Category breakdown
                st.markdown(f"""
                <div class="stats-container">
                    <strong>🎯 Query Analysis:</strong> Successfully processed your question | 
                    <strong>📚 Sources:</strong> {', '.join(categories)} | 
                    <strong>⚡ Response Time:</strong> Real-time | 
                    <strong>🔍 Relevance:</strong> High-precision results
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No relevant information found. Try rephrasing your question or exploring different keywords.")
        elif search_clicked and not query.strip():
            st.warning("⚠️ Please enter a question to search.")
        
        # Enhanced footer with animations
        st.markdown("---")
        st.markdown("""
        <div class="footer-text">
            <p>🚀 Powered by Pinecone Vector Search & OpenAI GPT-4 | 
            📚 Chai Docs Knowledge Base | 
            💡 Built with Streamlit</p>
            <p  style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8; text-color: #fff;">
            Experience the future of intelligent documentation search
            </p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit app"""
    try:
        app = ChaiDocsSearchApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
import streamlit as st
import os
from typing import List, Dict, Any
from datetime import datetime
import logging
import re
import json
# Langchain imports
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
# Pinecone
from pinecone import Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== API KEY CONFIGURATION =====

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Chai Docs Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
with open("style.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


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
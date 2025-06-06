import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
from langchain_pinecone import PineconeVectorStore

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import getpass
import os

from pinecone import Pinecone
from pinecone import ServerlessSpec

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Pinecone configuration
index_name = "chai-docs-index"

# Check for API keys
if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

class OptimizedChaiDocsLoader:
    def __init__(self, 
                 embedding_model_name: str = "text-embedding-3-large",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 index_name: str = "chai-docs-index",
                 max_workers: int = 10):
        
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.index_name = index_name
        self.max_workers = max_workers
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self._setup_pinecone_index()
        
        self.links = [
            'https://chaidocs.vercel.app/youtube/chai-aur-html/introduction/',
            'https://chaidocs.vercel.app/youtube/chai-aur-html/emmit-crash-course/',
            'https://chaidocs.vercel.app/youtube/chai-aur-html/html-tags/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/welcome/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/introduction/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/terminology/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/behind-the-scenes/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/branches/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/diff-stash-tags/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/managing-history/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/github/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/welcome/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/introduction/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/hello-world/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/variables-and-constants/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/data-types/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/operators/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/control-flow/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/loops/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/functions/',
            'https://chaidocs.vercel.app/youtube/chai-aur-django/welcome/',
            'https://chaidocs.vercel.app/youtube/chai-aur-django/getting-started/',
            'https://chaidocs.vercel.app/youtube/chai-aur-django/jinja-templates/',
            'https://chaidocs.vercel.app/youtube/chai-aur-django/tailwind/',
            'https://chaidocs.vercel.app/youtube/chai-aur-django/models/',
            'https://chaidocs.vercel.app/youtube/chai-aur-django/relationships-and-forms/',
            'https://chaidocs.vercel.app/youtube/chai-aur-sql/welcome/',
            'https://chaidocs.vercel.app/youtube/chai-aur-sql/introduction/',
            'https://chaidocs.vercel.app/youtube/chai-aur-sql/postgres/',
            'https://chaidocs.vercel.app/youtube/chai-aur-sql/normalization/',
            'https://chaidocs.vercel.app/youtube/chai-aur-sql/database-design-exercise/',
            'https://chaidocs.vercel.app/youtube/chai-aur-sql/joins-and-keys/',
            'https://chaidocs.vercel.app/youtube/chai-aur-sql/joins-exercise/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/welcome/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/setup-vpc/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/setup-nginx/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/nginx-rate-limiting/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/nginx-ssl-setup/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/node-nginx-vps/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/postgresql-docker/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/postgresql-vps/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/node-logger/'
        ]

    def _setup_pinecone_index(self):
        """Setup Pinecone index if it doesn't exist"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=3072,  # text-embedding-3-large dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws", 
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                time.sleep(10)
                logger.info(f"Index {self.index_name} created successfully")
            else:
                logger.info(f"Index {self.index_name} already exists")
                
            # Get index instance
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {str(e)}")
            raise

    def load_single_url(self, url: str) -> List[Document]:
        """Load a single URL and return documents with enhanced metadata"""
        try:
            loader = WebBaseLoader(web_path=url)
            docs = list(loader.lazy_load())
            
            if docs:
                doc = docs[0]
                # Enhanced metadata extraction
                enhanced_metadata = {
                    "source": url,
                    "title": doc.metadata.get("title", ""),
                    "description": doc.metadata.get("description", ""),
                    "language": doc.metadata.get("language", ""),
                    "content_length": len(doc.page_content),
                    "category": self._extract_category_from_url(url),
                    "topic": self._extract_topic_from_url(url),
                    "full_content": doc.page_content  # Store full content in metadata
                }
                
                # Update document metadata
                doc.metadata.update(enhanced_metadata)
                logger.info(f"Successfully loaded: {url}")
                return [doc]
            else:
                logger.warning(f"No content found for: {url}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading {url}: {str(e)}")
            return []

    def _extract_category_from_url(self, url: str) -> str:
        """Extract category from URL (e.g., 'html', 'git', 'django')"""
        parts = url.split('/')
        for part in parts:
            if part.startswith('chai-aur-'):
                return part.replace('chai-aur-', '')
        return "unknown"

    def _extract_topic_from_url(self, url: str) -> str:
        """Extract specific topic from URL"""
        parts = url.split('/')
        if len(parts) > 0:
            return parts[-2] if parts[-1] == '' else parts[-1]
        return "unknown"

    def load_all_documents(self) -> List[Document]:
        """Load all documents concurrently with progress tracking"""
        all_docs = []
        
        logger.info(f"Loading {len(self.links)} documents...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self.load_single_url, url): url 
                for url in self.links
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(self.links), desc="Loading documents") as pbar:
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        docs = future.result()
                        all_docs.extend(docs)
                        pbar.set_postfix({
                            "Current": url.split('/')[-2],
                            "Loaded": len(all_docs)
                        })
                    except Exception as e:
                        logger.error(f"Error processing {url}: {str(e)}")
                    finally:
                        pbar.update(1)
        
        logger.info(f"Successfully loaded {len(all_docs)} documents")
        return all_docs

    def process_and_store(self, batch_size: int = 50) -> PineconeVectorStore:
        """Load, chunk, and store all documents in Pinecone"""
        # Load all documents
        docs = self.load_all_documents()
        
        if not docs:
            raise ValueError("No documents were loaded successfully")

        # Display sample document info
        sample_doc = docs[0]
        logger.info("Sample document metadata:")
        for key, value in sample_doc.metadata.items():
            if key != "full_content":  # Don't print full content in logs
                logger.info(f"  {key}: {value}")

        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        split_docs = self.text_splitter.split_documents(docs)
        logger.info(f"Created {len(split_docs)} chunks from {len(docs)} documents")

        # Store in Pinecone in batches
        logger.info("Storing documents in Pinecone...")
        
        try:
            # Create vector store
            vector_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embedding_model
            )
            
            # Process in batches to avoid rate limits and memory issues
            for i in tqdm(range(0, len(split_docs), batch_size), desc="Storing in Pinecone"):
                batch = split_docs[i:i + batch_size]
                
                # Add documents to Pinecone
                vector_store.add_documents(batch)
                
                # Small delay to respect rate limits
                time.sleep(1)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(split_docs)-1)//batch_size + 1}")
            
            logger.info(f"Successfully stored all documents in Pinecone index '{self.index_name}'")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error storing documents in Pinecone: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded collection"""
        docs = self.load_all_documents()
        
        categories = {}
        total_content_length = 0
        
        for doc in docs:
            category = doc.metadata.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
            total_content_length += doc.metadata.get("content_length", 0)
        
        return {
            "total_documents": len(docs),
            "categories": categories,
            "total_content_length": total_content_length,
            "average_content_length": total_content_length / len(docs) if docs else 0
        }

    def get_pinecone_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            logger.error(f"Error getting Pinecone stats: {str(e)}")
            return {}

    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """Search documents using the vector store"""
        try:
            vector_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embedding_model
            )
            results = vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []


def main():
    """Main execution function"""
    try:
        # Initialize the loader
        loader = OptimizedChaiDocsLoader(
            chunk_size=1000,
            chunk_overlap=200,
            max_workers=8  # Adjust based on your system
        )
        
        # Get and display statistics
        stats = loader.get_collection_stats()
        logger.info("Collection Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Process and store all documents
        vector_store = loader.process_and_store(batch_size=30)
        
        logger.info("✅ All documents successfully processed and stored in Pinecone!")
        
        # Display Pinecone index stats
        pinecone_stats = loader.get_pinecone_stats()
        logger.info("Pinecone Index Statistics:")
        for key, value in pinecone_stats.items():
            logger.info(f"  {key}: {value}")
        
        # Optional: Test a sample query
        results = loader.search_documents("HTML tags", k=3)
        logger.info(f"Sample search returned {len(results)} results")
        
        if results:
            logger.info("Sample result:")
            logger.info(f"  Source: {results[0].metadata.get('source', 'N/A')}")
            logger.info(f"  Category: {results[0].metadata.get('category', 'N/A')}")
            logger.info(f"  Content preview: {results[0].page_content[:200]}...")
        
    except Exception as e:
        logger.error(f"❌ Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
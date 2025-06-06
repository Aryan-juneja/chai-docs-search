# 🔍 Chai Docs Intelligent Search

An AI-powered search engine for programming documentation that combines vector search with GPT-4 to provide comprehensive, contextual answers to your programming questions.

## ✨ Features

- **🤖 AI-Powered Responses**: Get detailed, contextual answers powered by OpenAI GPT-4
- **🔍 Vector Search**: Fast and accurate document retrieval using Pinecone vector database
- **📚 Multi-Category Support**: Search across HTML, Git, C Programming, Django, SQL, and DevOps documentation
- **🎯 Source References**: Every answer includes relevant source documents with links
- **📋 Search History**: Keep track of your recent searches
- **⚙️ Customizable Results**: Adjust the number of source documents retrieved
- **🎨 Beautiful UI**: Modern, responsive interface with dark theme
- **⚡ Real-time Search**: Instant results with comprehensive explanations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Pinecone API key
- OpenAI API key
- Streamlit

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aryan-juneja/chai-docs-search.git
   cd chai-docs-search
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys** (Choose one method)

   **Method 1: Environment Variables (Recommended)**
   ```bash
   export PINECONE_API_KEY="your-pinecone-api-key"
   export OPENAI_API_KEY="your-openai-api-key"
   ```

   **Method 2: Streamlit Secrets**
   Create `.streamlit/secrets.toml`:
   ```toml
   PINECONE_API_KEY = "your-pinecone-api-key"
   OPENAI_API_KEY = "your-openai-api-key"
   ```

   **Method 3: Environment File**
   Create `.env` file:
   ```env
   PINECONE_API_KEY=your-pinecone-api-key
   OPENAI_API_KEY=your-openai-api-key
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📦 Dependencies

```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-pinecone>=0.1.0
langchain-openai>=0.1.0
pinecone-client>=3.0.0
python-dotenv>=1.0.0
openai>=1.0.0
```

## 🏗️ Project Structure

```
chai-docs-search/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .env.example          # Environment variables template
├── .streamlit/
│   └── secrets.toml      # Streamlit secrets (optional)
└── data/                 # Data loading scripts (if any)
```

## 🔧 Configuration

### Pinecone Setup

1. Create a Pinecone account at [pinecone.io](https://pinecone.io)
2. Create a new index named `chai-docs-index`
3. Use the following settings:
   - **Dimension**: 3072 (for text-embedding-3-large)
   - **Metric**: cosine
   - **Cloud**: AWS (recommended)

### OpenAI Setup

1. Get your API key from [OpenAI Platform](https://platform.openai.com)
2. Ensure you have access to GPT-4 model
3. Set up billing if required

## 💡 Usage Examples

### Basic Search
```
How to create HTML forms?
```

### Advanced Queries
```
What are the differences between Django models and views?
How to set up Git branches for collaborative development?
Explain SQL JOIN operations with examples
```

### Code-Specific Questions
```
Show me Django model relationships examples
How to configure Nginx for production deployment?
What are Git stashing best practices?
```

## 🎯 Supported Categories

- **HTML & Web Development**: Forms, semantic tags, responsive design
- **Git & Version Control**: Branching, merging, history management
- **C Programming**: Fundamentals, data structures, algorithms
- **Django**: Models, views, templates, deployment
- **SQL**: Queries, joins, database design
- **DevOps**: Server configuration, deployment, containerization

## 🔍 How It Works

1. **Query Processing**: Your question is processed and converted to embeddings
2. **Vector Search**: Pinecone finds the most relevant documentation chunks
3. **Context Assembly**: Retrieved documents are formatted with metadata
4. **AI Generation**: GPT-4 generates a comprehensive answer using the context
5. **Response Display**: Answer is presented with source references

## 🛠️ Customization

### Modify Search Parameters
```python
# In app.py, adjust these settings:
num_results = st.slider("Number of source documents", 3, 10, 5)
temperature = 0.1  # LLM creativity (0.0 = deterministic, 1.0 = creative)
model = "gpt-4"    # OpenAI model selection
```

### Add New Categories
1. Update the categories list in `display_sidebar()`
2. Ensure your Pinecone index contains documents with matching category metadata

### Customize UI Theme
Modify the CSS variables in the Streamlit app for different color schemes and styling.

## 📊 Performance

- **Search Speed**: ~2-3 seconds per query
- **Accuracy**: High relevance through vector similarity
- **Scalability**: Handles thousands of documents efficiently
- **Cost**: Optimized token usage for cost-effective operation

## 🐛 Troubleshooting

### Common Issues

**"API Keys Missing" Error**
- Verify your API keys are correctly set
- Check environment variable names match exactly
- Ensure `.env` file is in the project root

**"Index not found" Error**
- Create the Pinecone index `chai-docs-index`
- Run data loading script to populate the index
- Verify index name matches in the code

**"No relevant documents found"**
- Try rephrasing your question
- Use more specific or general terms
- Check if your topic is covered in the documentation

**Slow Response Times**
- Reduce the number of source documents
- Check your internet connection
- Verify API rate limits

### Debug Mode

Enable debug logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black app.py

# Lint code
flake8 app.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Pinecone** for vector database infrastructure
- **OpenAI** for GPT-4 language model
- **Langchain** for LLM orchestration
- **Streamlit** for the web interface
- **Chai Docs** for the comprehensive programming documentation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/chai-docs-search/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/chai-docs-search/discussions)
- **Documentation**: [Wiki](https://github.com/your-username/chai-docs-search/wiki)

## 🔄 Version History

- **v1.0.0** - Initial release with basic search functionality
- **v1.1.0** - Added search history and improved UI
- **v1.2.0** - Enhanced error handling and performance optimization

---

<div align="center">

**🚀 Built with ❤️ for developers who love intelligent documentation search**

[⭐ Star this repo](https://github.com/your-username/chai-docs-search) • [🐛 Report Bug](https://github.com/your-username/chai-docs-search/issues) • [✨ Request Feature](https://github.com/your-username/chai-docs-search/issues)

</div>
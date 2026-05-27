cat > README.md << 'EOF'
# finance-rag
A RAG-based Q&A system that allows users to upload research reports and ask questions to quickly extract key insights.

## Features
- Extract key data and insights from complex research reports instantly
- Interactive Q&A format helps you retain information more effectively
- Converts dense charts, tables, and layouts into clear, readable text
- Cites source page numbers for every answer, so you can verify instantly

## Tech Stack
- **LangChain** — orchestrates the full pipeline (PDF loading, splitting, retrieval, Q&A)
- **Google Gemini API** — text embedding + LLM response generation
- **FAISS** — vector database for storing and retrieving relevant chunks
- **Streamlit** — web interface
- **PyMuPDF** — PDF parsing and text extraction

## How to Run

1. Clone the repository
```bash
   git clone https://github.com/vivioola/finance-rag.git
   cd finance-rag
```

2. Install dependencies
```bash
   pip install langchain langchain-google-genai langchain-community langchain-text-splitters faiss-cpu pymupdf streamlit python-dotenv google-generativeai
```

3. Add your API key — create a `.env` file in the project folder:
GOOGLE_API_KEY=your_key_here

4. Run the app
```bash
   streamlit run app.py
```
EOF

git add README.md
git commit -m "Add README"
git push


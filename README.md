# AI Document Analyzer

Professional Document Intelligence Platform powered by advanced AI.

## 🚀 Features

- **Multi-Document Support**: Load and analyze multiple PDFs, switch between them seamlessly.
- **Advanced AI Analysis**: Leveraging state-of-the-art language models for professional insights.
- **Smart Question Answering**: Ask complex questions and receive detailed, contextual answers with source citations.
- **Professional Summaries**: Generate comprehensive summaries and key insights from documents.
- **Career Analysis**: Specialized analysis for resumes and professional profiles.
- **Source Citations**: Every answer includes verifiable source references.

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **HuggingFace** (for embeddings)
- **ChromaDB** (in-memory vector store)
- **Mistral API** (for LLM-powered answers)

## ⚡️ Quick Start

1. **Clone the repository**
    ```sh
    git clone https://github.com/your-username/ai-docanalyzer.git
    cd ai-docanalyzer
    ```

2. **Create and activate a virtual environment**
    ```sh
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set environment variables**  
   Create a `.env` file in the project root with your [Mistral API key](https://docs.mistral.ai/):
    ```
    MISTRAL_API_KEY=your-mistral-api-key
    ```

5. **Run the app**
    ```sh
    streamlit run app.py
    ```

6. **Open in Browser**  
   Visit [http://localhost:8501](http://localhost:8501) to use the app.

## 📂 Project Structure

```
├── app.py
├── requirements.txt
├── README.md
├── .env.example
└── ...
```

## 📝 Usage

- **Upload PDF** documents for analysis.
- **Choose** a document and ask questions or use quick analysis buttons.
- **View answers** with references to document sources.

## 🔐 Security & Privacy

- Your documents are processed in memory. No persistent storage is used unless configured.
- API keys and sensitive data are managed via environment variables.

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

## 📄 License

[MIT License](LICENSE)

---

**Built with ❤️ by danish296**
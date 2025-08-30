# AI Document Analyzer

Professional Document Intelligence Platform powered by advanced AI.

---

## 🚀 Features

- **Innovative Multi-Document Support**: Seamlessly load, analyze, and switch between multiple PDF documents—empowering users to handle complex, multi-source analysis workflows.
- **Advanced AI-Powered Insights**: Designed with state-of-the-art language models for deep, meaningful document understanding and contextual analysis.
- **Smart Question Answering**: Ask any question about your documents and get detailed, well-referenced answers—backed up by transparent source citations.
- **Professional Summaries**: Instantly generate comprehensive, structured summaries and key insights from any document.
- **Career & Resume Analysis**: Specialized logic for extracting and analyzing professional profiles, resumes, and CVs.
- **Innovative User Interface**: Intuitively designed for clarity, efficiency, and a modern professional experience.
- **Source Citations**: Every answer includes verifiable source references for trust and transparency.

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** (UI)
- **LangChain** (document processing)
- **HuggingFace** (embeddings)
- **ChromaDB** (in-memory vector store)
- **Mistral API** (LLM-powered answers)

---

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

---

## 📂 Project Structure

```
├── app.py
├── requirements.txt
├── README.md
├── .env.example
└── ...
```

---

## 📝 Usage

- **Upload PDF** documents for instant, AI-powered analysis.
- **Select a document** and either ask custom questions or use quick analysis options.
- **View answers** with direct references to document sources.

---

## 🧠 Innovated & Designed For

This project is **innovated and designed** with a focus on:

- **Professionals and Analysts** needing fast, reliable insights from complex documents.
- **HR and Career Advisors** for in-depth resume/CV analysis.
- **Researchers, Students, and Teams** collaborating on document-heavy projects.
- Anyone looking for intelligent, trustworthy document understanding—delivered with cutting-edge AI.

---

## 🔐 Security & Privacy

- Documents are processed in memory (no persistent storage unless configured).
- API keys and sensitive data are managed via environment variables.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 📄 License

[MIT License](LICENSE)

---

**Innovated & Designed by danish296 | Built with ❤️ for professional intelligence**
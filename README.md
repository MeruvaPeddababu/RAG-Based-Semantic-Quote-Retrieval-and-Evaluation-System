# RAG-Based-Semantic-Quote-Retrieval-and-Evaluation-System


This project implements a Retrieval Augmented Generation (RAG) system to find relevant English quotes based on user queries. It then uses a large language model (Mistral via OpenRouter) to generate an answer or provide context based on the retrieved quote. The system also integrates RAGAS (RAG Assessment) for evaluating the performance of the RAG pipeline.

## Project Pipeline

1.  **Setup**:
    * Installs necessary Python packages including `gradio`, `sentence-transformers`, `faiss-cpu`, `datasets`, `pandas`, `numpy`, `requests`, and `ragas`.
2.  **Data Loading and Preprocessing**:
    * Loads a dataset of English quotes from HuggingFace (`Abirate/english_quotes`).
    * Performs basic cleaning by dropping rows with missing values in essential columns (`quote`, `author`, `tags`).
    * Creates a combined `text` field from the quote, author, and tags for richer semantic search.
3.  **Embedding and Indexing**:
    * Uses a pre-trained sentence transformer model (`all-MiniLM-L6-v2`) to generate dense vector embeddings for the quotes.
    * Builds a FAISS (Facebook AI Similarity Search) index (`IndexFlatL2`) for efficient similarity search over the quote embeddings.
4.  **Retrieval Augmented Generation (RAG)**:
    * **Retrieval**:
        * Takes a user's natural language query.
        * Encodes the query into an embedding using the same sentence transformer.
        * Searches the FAISS index to find the most similar quote(s) to the query.
    * **Generation**:
        * Constructs a prompt using the user's query and the retrieved quote (including its author and tags) as context.
        * Sends this prompt to a large language model (Mistral-7B Instruct, via the OpenRouter API) to generate a relevant answer.
5.  **Evaluation (RAGAS)**:
    * Uses the RAGAS framework to evaluate the RAG system's performance.
    * Metrics calculated include:
        * `faithfulness`: How well the generated answer is grounded in the provided context (retrieved quote).
        * `answer_relevancy`: How relevant the generated answer is to the original query.
        * `context_precision`: A measure of how relevant the retrieved context is to the query.
        * `context_recall`: A measure of the model's ability to retrieve all necessary context for the query.
6.  **User Interface**:
    * Implements a Gradio web interface that allows users to:
        * Enter a natural language query.
        * View the retrieved quote, its author, tags, the LLM-generated answer, and the RAGAS evaluation scores.

## Files

* `vijay_assign2.ipynb`: The Jupyter Notebook containing all the Python code for the system.

## Dependencies

* Python 3.x
* `gradio`
* `sentence-transformers`
* `faiss-cpu`
* `datasets`
* `pandas`
* `numpy`
* `requests`
* `ragas`
* `openai` (likely a dependency of `ragas` or other Langchain components)
* `torch` (dependency of `sentence-transformers`)
* `langchain`, `langchain-core`, `langchain-community`, `langchain_openai` (dependencies of `ragas`)

These can be installed via pip:
```bash
pip install gradio sentence-transformers faiss-cpu datasets pandas numpy requests ragas openai torch langchain langchain-core langchain-community langchain_openai

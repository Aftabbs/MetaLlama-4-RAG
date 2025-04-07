# Meta Llama 4 RAG Project

![image](https://github.com/user-attachments/assets/efb56687-7076-4e00-94e6-90b726230cc1)


This project integrates Meta's Llama 4 Scout model with a Retrieval-Augmented Generation (RAG) framework to enhance the model's performance by incorporating external knowledge sources. The application utilizes Faiss for vector database management, Gradio for the user interface, Groq for low-latency LLM inference, and Sentence Transformers for efficient text embeddings.

## Project Overview

- **Model**: [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- **Vector Database**: Faiss (CPU version)
- **User Interface**: Gradio
- **LLM Inference**: Groq
- **Sentence Embeddings**: Sentence Transformers (all-MiniLM model)

## Features

- **Retrieval-Augmented Generation (RAG)**: Enhances Llama 4 Scout's responses by retrieving relevant information from external datasets, improving accuracy and context relevance.
- **Efficient Vector Search**: Utilizes Faiss for fast and scalable similarity searches.
- **User-Friendly Interface**: Implements Gradio to provide an intuitive web-based UI for interactions.
- **Low-Latency Inference**: Leverages Groq's infrastructure for rapid LLM responses.
- **Compact Embeddings**: Employs Sentence Transformers' all-MiniLM model for efficient text representation.

## Model Comparison

| Feature               | Llama 4 Scout | GPT-4o        | Claude 3       | PaLM           |
|-----------------------|---------------|---------------|----------------|----------------|
| **Parameters**        | 17B active    | Not specified | Not specified  | Not specified  |
| **Architecture**      | MoE (16 experts) | Transformer  | Transformer    | Transformer    |
| **Context Length**    | 10M tokens    | Not specified | Not specified  | Not specified  |
| **Multimodal Support**| Yes           | Yes           | Yes            | Yes            |
| **Open Source**       | Yes           | No            | No             | No             |

*Note: The above comparison is based on available information and may not reflect the latest updates.*

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aftabbs/metallama4-rag.git
   cd meta-llama4-rag
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and Prepare Data**:
   - Ensure your dataset is in the `data/` directory.
   - Use the provided scripts to preprocess and index the data with Faiss.

4. **Configure API Keys**:
   - Set up your Groq API key in the environment variables:
     ```bash
     export GROQ_API_KEY='your_api_key'
     ```

5. **Run the Application**:
   ```bash
   python app.py
   ```



6. **Access the Interface**:
   - Open your browser and navigate to `http://localhost:7860` to interact with the application.

## References

- [Meta AI Blog: Llama 4 Multimodal Intelligence](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Faiss: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
- [Gradio: Build Machine Learning Web Apps](https://gradio.app/)
- [Groq: Accelerating AI Inference](https://groq.com/)
- [Sentence Transformers Documentation](https://www.sbert.net/)

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

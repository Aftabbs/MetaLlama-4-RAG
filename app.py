import gradio as gr
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer   
from groq import Groq   
import os  
 
client = Groq(api_key=os.environ['GROQ_API_KEY'])


def extract_text_from_pdf(pdf_input):

    text = ""
    if isinstance(pdf_input, dict):
        pdf_path = pdf_input.get("name", None)
    else:
        pdf_path = pdf_input

    with pdfplumber.open(pdf_input) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def chunk_text(text, chunk_size=500):
    """
    Chunk the text into smaller pieces.
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def index_chunks(chunks):
    """
    Create embeddings for each text chunk and build a FAISS index.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve_chunks(question, chunks, index, embeddings, top_k=3):
    """
    Retrieve the most relevant chunks for the question.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    question_embedding = model.encode([question])
    distances, indices = index.search(question_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def generate_answer(question, chunks):
    """
    Generate an answer using the Groq chat completion API based on the provided context.
    """
    context = "\n".join(chunks)
    messages = [
        {"role": "system", "content": "Use the given context to answer accurately."},
        {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
    ]
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.2,
        top_p=0.1,
        max_tokens=1000,
        stream=True
    )
    answer = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            answer += chunk.choices[0].delta.content
    return answer

def process_pdf_and_query(pdf_input, question):

    text = extract_text_from_pdf(pdf_input)
    if not text.strip():
        return "Error: No text could be extracted from the PDF. Please check the file."

    chunks = chunk_text(text)

    index, embeddings = index_chunks(chunks)

    relevant_chunks = retrieve_chunks(question, chunks, index, embeddings)

    answer = generate_answer(question, relevant_chunks)
    return answer


custom_css = """
body {background-color: #f5f5f5;}
.gradio-container {max-width: 800px; margin: auto; padding: 20px;}
.title {text-align: center; font-size: 2em; margin-bottom: 10px;}
.description {text-align: center; margin-bottom: 20px; font-style: italic; color: #555;}
"""

header_image = "/content/55571654-d64c-4d4f-aee7-a516a5d9949e.png?text=DeepSeek+RAG"

with gr.Blocks(css=custom_css, title="Meta Llama 4 RAG") as demo:
    with gr.Row():
        gr.Markdown("<div class='title'>Llama 4 Scout</div>")
    with gr.Row():
        gr.Image(header_image, elem_id="header_image")
    with gr.Row():
        gr.Markdown("<div class='description'>Upload your PDF and ask a question to retrieve answers based on the document content.</div>")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], file_count="single", interactive=True)
    with gr.Row():
        query_input = gr.Textbox(label="Enter your query", placeholder="Type your question here...", lines=2)

    with gr.Row():
        process_button = gr.Button("Get Answer")

    answer_output = gr.Textbox(label="Answer", interactive=False)

    process_button.click(
        fn=process_pdf_and_query,
        inputs=[pdf_input, query_input],
        outputs=answer_output
    )

demo.launch()

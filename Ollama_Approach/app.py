import streamlit as st
import torch
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template
from transformers import CLIPProcessor, CLIPModel
import os
import tempfile
import io
from PIL import Image
import fitz

# Initialize CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_pdf_tables(pdf_docs):
    tables = []
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_file:
            for page in pdf_file.pages:
                tables.append(page.extract_tables())
    return tables

def get_images_with_text(pdf_docs, text_range=1000):
    images_with_text = []

    for pdf in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf.read())
            temp_pdf_path = temp_pdf.name

        pdf_reader = fitz.open(temp_pdf_path)
        for page_num in range(len(pdf_reader)):
            page = pdf_reader[page_num]
            text = page.get_text("text")
            image_list = page.get_images(full=True)

            for img in image_list:
                xref = img[0]
                base_image = pdf_reader.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))

                # Get the image bounding box and surrounding text
                bbox = fitz.Rect(img[1], img[2], img[3], img[4])
                bbox_text = get_surrounding_text(page, bbox, text, text_range)
                images_with_text.append({
                    'image': image,
                    'text': bbox_text,
                    'page_num': page_num
                })

        os.remove(temp_pdf_path)

    # Store the images with text in local folder and corresponding surrounding text in another folder to view later if needed
    os.makedirs('images', exist_ok=True)
    os.makedirs('texts', exist_ok=True)
    for i, item in enumerate(images_with_text):
        image_path = f"images/image_{i}.png"
        text_path = f"texts/text_{i}.txt"
        item['image'].save(image_path)
        with open(text_path, 'w') as f:
            f.write(item['text'])

    return images_with_text

def get_surrounding_text(page, bbox, full_text, text_range):
    words = page.get_text("words")  # Get individual words on the page
    surrounding_text = []
    for word in words:
        word_bbox = fitz.Rect(word[:4])
        if word_bbox.intersects(bbox):
            surrounding_text.append(word[4])
        # Check for proximity
        elif word_bbox.distance_to(bbox) < text_range:
            surrounding_text.append(word[4])

    surrounding_text = ' '.join(surrounding_text)
    return surrounding_text

def find_most_similar_image(clip_model, clip_processor, answer, images_with_text):
    answer_inputs = clip_processor(text=[answer], return_tensors="pt", padding=True)
    answer_embedding = clip_model.get_text_features(**answer_inputs)

    max_similarity = -1
    best_match = None

    for item in images_with_text:
        image = item['image']
        image_inputs = clip_processor(images=image, return_tensors="pt", padding=True)
        image_embedding = clip_model.get_image_features(**image_inputs)

        similarity = torch.nn.functional.cosine_similarity(answer_embedding, image_embedding).item()

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = item

    return best_match

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        length_function=len,
        separators=[" ", ",", "\n", "."]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, tables):
    table_texts = []
    for table in tables:
        for row in table:
            if all(isinstance(cell, list) for cell in row):
                row = [item for sublist in row for item in sublist]
            row = [item for item in row if item is not None]
            table_texts.append(' '.join(row))

    all_texts = text_chunks + table_texts

    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(texts=all_texts, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = Ollama(model="llama3")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    initial_instruction = (
        "You are an AI Assistant that follows instructions extremely well. Please be truthful and give direct answers. If you do not know the answer or if the user query is not in context, please say 'I don't know'. "
        "You will lose the job if you answer out-of-context questions. If a query is confusing or unclear, ask a probing question to clarify. Remember to only return the AI's answer."
    )
    initial_response = "OK."
    
    memory.save_context({"user": initial_instruction},{"response": initial_response})
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i == 0 or i == 1:
            continue

        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

            # Display relevant images
            images_with_text = st.session_state.images_with_text
            best_match = find_most_similar_image(clip_model, clip_processor, message.content, images_with_text)
            
            if best_match:
                st.image(best_match['image'], caption='Relevant Image', use_column_width=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "images_with_text" not in st.session_state:
        st.session_state.images_with_text = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                tables = get_pdf_tables(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                
                images_with_text = get_images_with_text(pdf_docs)

                # Store images with text in session state
                st.session_state.images_with_text = images_with_text

                vectorstore = get_vectorstore(text_chunks, tables)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()

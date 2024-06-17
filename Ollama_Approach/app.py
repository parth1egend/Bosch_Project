import streamlit as st
import torch
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template
from sentence_transformers import SentenceTransformer, util
import os
import tempfile
import fitz  # PyMuPDF
import io
from PIL import Image, ExifTags, ImageOps, UnidentifiedImageError

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

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        length_function=len,
        separators=[" ", ",", "\n", "."]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_images_from_pdf(pdf_docs):
    pages_data = []
    for pdf in pdf_docs:
        pdf_path = os.path.join(tempfile.gettempdir(), pdf.name)
        with open(pdf_path, 'wb') as f:
            f.write(pdf.getbuffer())
        pdf_document = fitz.open(pdf_path)
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text = page.get_text()
            images = []
            for image_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                try:
                    image = Image.open(io.BytesIO(base_image["image"]))
                    
                    # Rotate image if necessary
                    image = ImageOps.exif_transpose(image)
                    
                    image_path = os.path.join(tempfile.gettempdir(), f"{pdf.name}_image_{page_number}_{image_index}.jpg")
                    image.save(image_path)
                    images.append(image_path)
                except (UnidentifiedImageError, KeyError) as e:
                    print(f"Error processing image on page {page_number}, index {image_index}: {e}")
            pages_data.append({
                "pdf_name": pdf.name,
                "page_number": page_number,
                "text": text,
                "images": images
            })
    return pages_data

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


def detect_ambiguity(user_question, pdf_docs):
    car_keywords = [
        "car", "warranty", "model", "brand", "mileage", "engine", "repair", 
        "maintenance", "service", "accident", "VIN", "SUV","ESC", "sedan", "coupe",
        "hatchback", "convertible", "hybrid", "electric", "fuel", "performance",
        "safety", "parts", "accessories", "recall", "insurance",

        # Driving experience
        "drive", "handling", "comfort", "technology", "features", "interior",
        "exterior", "seating", "cargo space", "ride quality", "noise level",

        # Buying and selling
        "buy", "sell", "lease", "trade-in", "financing", "cost", "price", "value",
        "dealership", "used car", "new car", "certified pre-owned", "private seller",

        # Colors and materials
        "color", "paint", "leather", "fabric", "interior trim", "exterior trim",

        # Performance related
        "horsepower", "torque", "acceleration", "top speed", "fuel efficiency",
        "mpg", "range", "transmission", "drivetrain", "suspension", "brakes",

        # Safety features
        "airbags", "anti-lock brakes (ABS)", "traction control", "electronic stability control",
        "lane departure warning", "blind spot monitoring", "forward collision warning",
        "automatic emergency braking", "rearview camera", "parking sensors",

        # Technology features
        "infotainment system", "navigation", "touchscreen", "smartphone integration",
        "Bluetooth", "Wi-Fi", "USB ports", "sunroof", "heated seats", "ventilated seats",

        # Car types (more specific)
        "muscle car", "sports car", "luxury car", "off-road vehicle", "truck",
        "minivan", "wagon", "compact car", "subcompact car", "midsize car",

        # Maintenance and repair (more specific)
        "oil change", "tire rotation", "brake pads", "spark plugs", "battery",
        "air filter", "cabin air filter", "transmission fluid", "coolant", "diagnostics",

        # Legal and documentation
        "registration", "title", "emissions test", "owner's manual", "service history"
    ]
    car_names = [# Brands (more specific) in small letters
        "audi", "bmw", "buick", "cadillac", "chevrolet", "chrysler", "dodge", "fiat",
        "ford", "gmc", "honda", "hyundai", "infiniti", "jaguar", "jeep", "kia",
        "land rover", "lexus", "lincoln", "mazda", "mercedes-benz", "mini", "mitsubishi",
        "nissan", "porsche", "ram", "subaru", "tesla", "toyota", "volkswagen", "volvo",
        "nissan","hyundai","ford","toyota","honda","chevrolet","bmw","audi","mercedes-benz",
        "tata","mahindra","maruti","suzuki","renault","volkswagen","skoda","fiat","jeep",
        "kia","kia","mg","volvo","land rover","jaguar","porsche","lamborghini","ferrari",
        ]
    # If the question has any of the car name then no ambiguity
    if any(keyword in user_question.lower() for keyword in car_names):
        return False
    
    if any(keyword in user_question.lower() for keyword in car_keywords) and len(pdf_docs) > 1:
        return True
    return False

# def handle_userinput(user_question, show_highest_similarity, similarity_threshold, num_images):
    
    
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']
    
#     llm_response = response['answer']

#     # Store the current question and response
#     current_conversation = {"user": user_question, "bot": llm_response, "images": []}

#     # Display the current question and response
#     st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
#     st.write(bot_template.replace("{{MSG}}", llm_response), unsafe_allow_html=True)

#     # Check similarity between the response and text in pages_data
#     embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#     response_embedding = embeddings.encode(llm_response, convert_to_tensor=True)
    
#     similarities = []
    
#     for page_data in st.session_state.pages_data:
#         page_text_embedding = embeddings.encode(page_data['text'], convert_to_tensor=True)
#         similarity = util.pytorch_cos_sim(response_embedding, page_text_embedding).item()
#         similarities.append((similarity, page_data))
    
#     # Sort pages by similarity in descending order
#     similarities.sort(key=lambda x: x[0], reverse=True)
    
#     is_image_displayed = False
#     displayed_images = []
#     # Display images based on user selection
#     if show_highest_similarity:
#         for similarity, page_data in similarities:
#             if page_data['images']:
#                 st.write(f"Related images from PDF '{page_data['pdf_name']}', page {page_data['page_number']} with similarity {similarity:.2f}:")
#                 is_image_displayed = True
#                 images_threshold = min(num_images, len(page_data['images']))
#                 for image_path in page_data['images']:
#                     st.image(image_path)   
#                     displayed_images.append(image_path)
#                     images_threshold -= 1
#                     if images_threshold == 0:
#                         break
#                 break
#     else:
#         for similarity, page_data in similarities:
#             if similarity >= similarity_threshold and page_data['images']:
#                 st.write(f"Related images from PDF '{page_data['pdf_name']}', page {page_data['page_number']} with similarity {similarity:.2f}:")
#                 is_image_displayed = True
#                 images_threshold = min(num_images, len(page_data['images']))
#                 for image_path in page_data['images']:
#                     st.image(image_path)
#                     displayed_images.append(image_path)
#                     images_threshold -= 1
#                     if images_threshold == 0:
#                         break
    
#     if not is_image_displayed:
#         st.write(f"No images with the given similarity threshold {similarity_threshold:.2f} found.")

#     # Add the displayed images to the current conversation
#     current_conversation["images"] = displayed_images

#     # Append the current conversation to the chat history
#     st.session_state.full_chat_history.append(current_conversation)


def handle_userinput(user_question, show_highest_similarity, similarity_threshold, num_images,pdf_docs):
        
    response = st.session_state.conversation({'question': user_question})
    
    if detect_ambiguity(user_question, pdf_docs):
        response['answer'] = "Can you please specify which car or pdf you are referring to?"
    
    
    st.session_state.chat_history = response['chat_history']
    llm_response = response['answer']
    

    # Store the current question and response
    current_conversation = {"user": user_question, "bot": llm_response, "images": []}

    # Display the current question and response
    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", llm_response), unsafe_allow_html=True)

    # Check similarity between the response and text in pages_data
    embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    response_embedding = embeddings.encode(llm_response, convert_to_tensor=True)
    
    similarities = []
    
    for page_data in st.session_state.pages_data:
        page_text_embedding = embeddings.encode(page_data['text'], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(response_embedding, page_text_embedding).item()
        similarities.append((similarity, page_data))
    
    # Sort pages by similarity in descending order
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    is_image_displayed = False
    displayed_images = []
    # Display images based on user selection
    if show_highest_similarity:
        for similarity, page_data in similarities:
            if page_data['images']:
                st.write(f"Related images from PDF '{page_data['pdf_name']}', page {page_data['page_number']} with similarity {similarity:.2f}:")
                is_image_displayed = True
                images_threshold = min(num_images, len(page_data['images']))
                for image_path in page_data['images']:
                    st.image(image_path)   
                    displayed_images.append(image_path)
                    images_threshold -= 1
                    if images_threshold == 0:
                        break
                break
    else:
        for similarity, page_data in similarities:
            if similarity >= similarity_threshold and page_data['images']:
                st.write(f"Related images from PDF '{page_data['pdf_name']}', page {page_data['page_number']} with similarity {similarity:.2f}:")
                is_image_displayed = True
                images_threshold = min(num_images, len(page_data['images']))
                for image_path in page_data['images']:
                    st.image(image_path)
                    displayed_images.append(image_path)
                    images_threshold -= 1
                    if images_threshold == 0:
                        break
    
    if not is_image_displayed:
        st.write(f"No images with the given similarity threshold {similarity_threshold:.2f} found.")

    # Add the displayed images to the current conversation
    current_conversation["images"] = displayed_images

    # Append the current conversation to the chat history
    st.session_state.full_chat_history.append(current_conversation)
        


def display_full_chat_history():
    # Display the full chat history
    for conversation in st.session_state.full_chat_history:
        st.write(user_template.replace("{{MSG}}", conversation["user"]), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", conversation["bot"]), unsafe_allow_html=True)
        if conversation["images"]:
            for image_path in conversation["images"]:
                st.image(image_path)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pages_data" not in st.session_state:
        st.session_state.pages_data = []
    if "full_chat_history" not in st.session_state:
        st.session_state.full_chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                tables = get_pdf_tables(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                
                st.session_state.pages_data = get_images_from_pdf(pdf_docs)

                vectorstore = get_vectorstore(text_chunks, tables)
                st.session_state.conversation = get_conversation_chain(vectorstore)

        st.subheader("Image Display Options")
        show_highest_similarity = st.checkbox("Show images from the page with highest similarity", value=True)
        if not show_highest_similarity:
            similarity_threshold = st.slider("Images would be displayed from the pages with more than below similarity", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        else:
            similarity_threshold = None
            
        st.subheader("Number of Images to Display")
        num_images = st.slider("Set number of images to display per page of the PDF", min_value=1, max_value=10, value=5, step=1)

    if st.button("Submit"):
        handle_userinput(user_question, show_highest_similarity, similarity_threshold, num_images,pdf_docs)
    
    if st.button("Show Full Chat History"):
        display_full_chat_history()

if __name__ == '__main__':
    main()


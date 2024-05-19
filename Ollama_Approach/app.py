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

def get_vectorstore(text_chunks, tables):
    # Convert tables into a list of strings
    table_texts = []
    for table in tables:
        for row in table:
            # Flatten the row if it's a list of lists
            if all(isinstance(cell, list) for cell in row):
                row = [item for sublist in row for item in sublist]
            # Filter out None values
            row = [item for item in row if item is not None]
            table_texts.append(' '.join(row))

    # Combine text_chunks and table_texts
    all_texts = text_chunks + table_texts

    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': device}, encode_kwargs={'device': device})
    
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
        if i==0 or i==1 : continue

        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

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
                vectorstore = get_vectorstore(text_chunks, tables)
                print("Vector store created.")
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()

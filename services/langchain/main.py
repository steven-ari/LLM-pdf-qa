import requests
from dotenv import load_dotenv

from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import streamlit as st

# Load environment variables
load_dotenv('.env')

def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)

def main():
    st.title("Chat with your PDF 💬")
    
    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # Text variable will store the pdf text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split the text into chunks using langchain
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        vectordb = Chroma.from_texts(
            chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory='./data'
        )
        vectordb.persist()

        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
            return_source_documents=True
        )
        
        query = st.text_input('Ask your question just like in ChatGPT')
        cancel_button = st.button('Cancel')
        
        if cancel_button:
            st.stop()
        
        if query:
            with get_openai_callback() as cost:
                result = qa_chain({'query': query})
                print(cost)
                
            st.write(result['result'])
    
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.title("OR chat with your Link 💬")
    url = st.text_input("or enter a URL and chat with it")
    
    if url:
        text_url = extract_text_from(url)

        # Split the text into chunks using langchain
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks_url = text_splitter.split_text(text_url)
        print(chunks_url)
        
        vectordb_url = Chroma.from_texts(
            chunks_url,
            embedding=OpenAIEmbeddings(),
            persist_directory='./data'
        )
        vectordb_url.persist()

        qa_chain_url = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            retriever=vectordb_url.as_retriever(search_kwargs={'k': 6}),
            return_source_documents=True
        )
        
        query_url = st.text_input('Ask your question just like in ChatGPT to the html')
        cancel_button_url = st.button('Cancel url')
        
        if cancel_button_url:
            st.stop()
        
        if query_url:
            with get_openai_callback() as cost:
                result_url = qa_chain_url({'query': query_url})
                print(cost)
                
            st.write(result_url['result'])
            
            
if __name__ == "__main__":
    main()

from dotenv import load_dotenv

from PyPDF2 import PdfReader
import streamlit as st
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

# Load environment variables
load_dotenv('.env')

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
            
            
if __name__ == "__main__":
    main()

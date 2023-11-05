from urllib.parse import urlparse

import requests
import streamlit as st
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load environment variables
load_dotenv('.env')


def extract_text_from(url: str) -> str:
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)


def process_text(text: str, db_dir: str) -> BaseRetrievalQA:
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
        persist_directory=db_dir
    )
    vectordb.persist()

    return RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
        return_source_documents=True
    )


def is_valid_url(text: str) -> bool:
    try:
        result = urlparse(text)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def main():
    st.title("Chat with your PDF 💬")

    pdf = st.file_uploader('Upload your PDF Document, previous uploads will be saved and emptied daily', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = "\n".join(page.extract_text() for page in pdf_reader.pages)
        qa_chain = process_text(text, './data')

        query = st.text_input('Ask your question just like in ChatGPT')
        cancel_button = st.button('Cancel')

        if cancel_button:
            st.stop()

        if query:
            with get_openai_callback() as cost:
                result = qa_chain({'query': query})
                print(cost)

            st.write(result['result'])

    for _ in range(7):
        st.write("")

    st.header('OR.. chat with your link 🔗 ', divider='rainbow')
    url = st.text_input("Enter a URL and chat with it, previous links will be saved and emptied daily")

    if is_valid_url(url):
        text_url = extract_text_from(url)
        qa_chain_url = process_text(text_url, './data_url')

        query_url = st.text_input('Ask your question just like in ChatGPT to the html')
        cancel_button_url = st.button('Cancel url')

        if cancel_button_url:
            st.stop()

        if query_url:
            with get_openai_callback() as cost:
                result_url = qa_chain_url({'query': query_url})
                print(cost)

            st.write(result_url['result'])
    else:
        st.write('Give only valid url')


if __name__ == "__main__":
    main()

import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
import os

st.title("Web Content Q&A")

url = st.text_input("Enter URL:")
question = st.text_input("Enter your question:")

if st.button("Get Answer") and url and question:
    with st.spinner("Processing..."):
        try:
            def get_sub_urls_with_metadata(base_url):
                response = requests.get(base_url)
                soup = BeautifulSoup(response.text, "html.parser")
                results = []
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    full_url = urljoin(base_url, href)
                    if full_url.startswith("http"):
                        anchor_text = a_tag.get_text(strip=True)
                        parsed_url = urlparse(full_url)
                        last_part = unquote(parsed_url.fragment or parsed_url.path.rstrip("/").split("/")[-1])
                        full_text = anchor_text
                        if last_part and last_part.lower() not in anchor_text.lower():
                            full_text += f" ({last_part})"
                        results.append({"url": full_url, "text": full_text})
                return results

            sublinks = get_sub_urls_with_metadata(url)
            model = SentenceTransformer('all-MiniLM-L6-v2')
            link_texts = [item["text"] for item in sublinks]
            link_embeddings = model.encode(link_texts, convert_to_tensor=True)
            query_embedding = model.encode(question, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, link_embeddings)[0]
            top_indices = cos_scores.topk(k=3).indices
            selected_urls = {sublinks[idx]['url'] for idx in top_indices}

            loader = WebBaseLoader(list(selected_urls))
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)

            embedding_model = HuggingFaceEmbeddings()
            faiss_index = FAISS.from_documents(documents=chunks, embedding=embedding_model)

            llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=faiss_index.as_retriever())
            result = chain({"question": question})

            st.success(result['answer'])
            st.info(f"Sources: {result['sources']}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
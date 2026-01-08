import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import time

# Page configuration
st.set_page_config(page_title="Web Content Q&A", page_icon="🔍", layout="wide")

st.title("🔍 Web Content Q&A System")
st.markdown("Enter a URL and ask questions about its content")

# Input fields
col1, col2 = st.columns([2, 1])
with col1:
    url = st.text_input("Enter URL:", placeholder="https://example.com")
with col2:
    top_k = st.slider("Number of links to analyze:", 1, 5, 3)

question = st.text_area("Enter your question:", placeholder="What is this page about?")

# API Key input (optional - can be set in environment)
if not os.getenv("GROQ_API_KEY"):
    groq_api_key = st.text_input("Groq API Key:", type="password", 
                                  help="Get your API key from https://console.groq.com")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize session state for caching
if 'cached_url' not in st.session_state:
    st.session_state.cached_url = None
if 'cached_faiss' not in st.session_state:
    st.session_state.cached_faiss = None


@st.cache_data
def get_sub_urls_with_metadata(base_url, timeout=10):
    """Extract all links from the base URL with metadata"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(base_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        results = []
        base_domain = urlparse(base_url).netloc
        
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(base_url, href)
            
            # Filter to same domain and valid URLs
            if full_url.startswith("http") and urlparse(full_url).netloc == base_domain:
                anchor_text = a_tag.get_text(strip=True)
                parsed_url = urlparse(full_url)
                last_part = unquote(parsed_url.path.rstrip("/").split("/")[-1])
                
                # Create descriptive text
                full_text = anchor_text if anchor_text else last_part
                if last_part and anchor_text and last_part.lower() not in anchor_text.lower():
                    full_text += f" ({last_part})"
                
                if full_text:  # Only add if there's meaningful text
                    results.append({"url": full_url, "text": full_text})
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for item in results:
            if item["url"] not in seen:
                seen.add(item["url"])
                unique_results.append(item)
        
        return unique_results
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return []


@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')


def find_relevant_links(sublinks, question, model, k=3):
    """Find most relevant links using semantic similarity"""
    if not sublinks:
        return []
    
    link_texts = [item["text"] for item in sublinks]
    link_embeddings = model.encode(link_texts, convert_to_tensor=True)
    query_embedding = model.encode(question, convert_to_tensor=True)
    
    cos_scores = util.pytorch_cos_sim(query_embedding, link_embeddings)[0]
    top_k = min(k, len(sublinks))
    top_indices = cos_scores.topk(k=top_k).indices
    
    selected_links = [sublinks[idx] for idx in top_indices]
    return selected_links


def create_vector_store(urls):
    """Create FAISS vector store from URLs"""
    try:
        loader = WebBaseLoader(urls)
        loader.requests_kwargs = {'verify': True, 'timeout': 10}
        documents = loader.load()
        
        if not documents:
            st.error("No content could be loaded from the URLs")
            return None
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and FAISS index
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        faiss_index = FAISS.from_documents(documents=chunks, embedding=embedding_model)
        
        return faiss_index, embedding_model
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


def get_answer(question, faiss_index):
    """Get answer using RAG chain"""
    try:
        # Create prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer in detail:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Initialize LLM
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024
        )
        
        # Create retrieval chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=faiss_index.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        result = chain({"query": question})
        return result
    except Exception as e:
        st.error(f"Error getting answer: {str(e)}")
        return None


# Main execution
if st.button("🚀 Get Answer", type="primary") and url and question:
    if not os.getenv("GROQ_API_KEY"):
        st.error("Please provide a Groq API key")
        st.stop()
    
    with st.spinner("Processing your request..."):
        try:
            # Step 1: Extract sublinks
            with st.status("Extracting links from page...") as status:
                sublinks = get_sub_urls_with_metadata(url)
                st.write(f"Found {len(sublinks)} links")
                status.update(label="Links extracted!", state="complete")
            
            if not sublinks:
                st.warning("No sublinks found. Analyzing main page only...")
                selected_urls = [url]
                selected_links = [{"url": url, "text": "Main page"}]
            else:
                # Step 2: Find relevant links
                with st.status("Finding relevant content...") as status:
                    model = load_embedding_model()
                    selected_links = find_relevant_links(sublinks, question, model, k=top_k)
                    selected_urls = [link['url'] for link in selected_links]
                    st.write(f"Selected {len(selected_urls)} most relevant pages")
                    status.update(label="Relevant content identified!", state="complete")
            
            # Step 3: Create vector store
            with st.status("Processing content...") as status:
                result = create_vector_store(selected_urls)
                if result is None:
                    st.stop()
                faiss_index, embedding_model = result
                st.write("Content indexed successfully")
                status.update(label="Content processed!", state="complete")
            
            # Step 4: Get answer
            with st.status("Generating answer...") as status:
                answer_result = get_answer(question, faiss_index)
                if answer_result:
                    st.write("Answer generated")
                    status.update(label="Complete!", state="complete")
            
            if answer_result:
                # Display results
                st.markdown("---")
                st.subheader("Answer")
                st.write(answer_result['result'])
                
                st.markdown("---")
                st.subheader("Analyzed Pages")
                for i, link in enumerate(selected_links, 1):
                    st.markdown(f"{i}. [{link['text']}]({link['url']})")
                
                # Show source documents
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(answer_result['source_documents'][:3], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content[:500] + "...")
                        st.markdown(f"*From: {doc.metadata.get('source', 'Unknown')}*")
                        st.markdown("---")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

# Sidebar with instructions
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    1. **Enter a URL** of the webpage you want to analyze
    2. **Ask a question** about the content
    3. **Set number of links** to analyze (1-5)
    4. **Click 'Get Answer'** to process
    
    The system will:
    - Extract links from the page
    - Find the most relevant pages
    - Analyze the content
    - Provide an answer with sources
    """)
    
    st.markdown("---")
    st.markdown("**Note:** You need a [Groq API key](https://console.groq.com) to use this app.")

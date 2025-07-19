# Web Content Q&A with LangChain, FAISS, and Groq

This project is a Streamlit web app that allows users to enter any URL and ask questions about its content. The app fetches related subpages, selects the most relevant ones using semantic similarity, loads and processes their content, and answers the question using a large language model.

## Features

- Extracts sub-links from a web page
- Ranks links using semantic similarity (SentenceTransformer)
- Loads content using WebBaseLoader from LangChain
- Splits text with RecursiveCharacterTextSplitter
- Stores and retrieves chunks using FAISS
- Answers questions using Groq LLM (`llama3-70b-versatile`)
- Displays sources for the generated answer
<img width="1470" alt="Screenshot 2025-06-09 at 8 52 48 PM" src="https://github.com/user-attachments/assets/db2ca71f-4d34-4ea7-8212-a1aaa319ffe0" />

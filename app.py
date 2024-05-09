#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
from docx import Document
import os
import openai


# In[4]:


import os

# In[5]:


chat_history = []


# In[6]:


def get_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text

def get_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def get_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    text = ""
    for index, row in df.iterrows():
        row_text = ' | '.join(str(x) for x in row if pd.notna(x)) + "\n"
        text += row_text
        #print(row)
    return text


# In[7]:


def get_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    text = ""
    for index, row in df.iterrows():
        row_text = ' | '.join(str(x) for x in row if pd.notna(x)) + "\n"
        text += row_text
        #print(row)
    return text
#print(get_text_from_csv("C:/Users/nicon/Desktop/專利檢索/專利資料/專利100.csv"))


# In[ ]:


def generate_response(user_input, files_folder):
    text = ""
    for filename in os.listdir(files_folder):
        file_path = os.path.join(files_folder, filename)
        if filename.endswith('.pdf'):
            text += get_text_from_pdf(file_path)
        elif filename.endswith('.docx'):
            text += get_text_from_docx(file_path)
        elif filename.endswith('.csv'):
            text += get_text_from_csv(file_path)
            #print(text)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2500,
        chunk_overlap=400,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Instantiate the Embedding Model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.environ['OPENAI_API_KEY'])

    # Create Index- Load document chunks into the vectorstore
    faiss_vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
    )
    
    # Create a retriever
    retriever = faiss_vectorstore.as_retriever()

    # Use the invoke method to execute the search
    docs = retriever.invoke(user_input, top_k=5)
    #print(docs)

    llm = ChatOpenAI(
        model_name="gpt-4-1106-preview",
        temperature=0.4
    )

    chain = load_qa_chain(llm, chain_type="stuff")

    with get_openai_callback() as cb:
           response = chain.invoke({"input_documents": docs,"question":user_input}, return_only_outputs=True)
    response = {'response': response}

    chat_history.append({'user': user_input, 'assistant': response['response']})

    return response


# Main conversation loop
files_folder = 'C:/Users/nicon/Desktop/專利檢索/專利資料'
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = generate_response(user_input, files_folder)
    print("AI:", response['response'])


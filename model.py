import streamlit as st
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import PyPDF2
import re

GOOGLE_API_KEY = "AIzaSyCVDoR3U-WDV-_DmeXlt76ubLeNTOw2n64"
genai.configure(api_key=GOOGLE_API_KEY)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
        return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
get_vector_store(get_text_chunks(extract_text_from_pdf('iesc111.pdf')))
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context,if the user asks about the topics or subtopics try to find the paragraph headings
    and organize them, understand mathematical formulas and their context , if the user asks a specific portion of the context find it and summarize it so that it can be understood easily  
    find the answer from the context, make sure to provide all the details, and don't provide the wrong answer.\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    embedded_question = embeddings.embed_query(user_question)
    docs = new_db.similarity_search_by_vector(embedded_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response

def summarize_text(text):
    prompt_template = """
    Summarize the following text:\n\n
    Text:\n{text}\n
    Summary:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    summary = model({"text": text})
    return summary['output_text']

def agent_decision(user_question):

    if re.match(r'^(hi|hello|hey|greetings)', user_question, re.I):
        return "greeting"

    if re.search(r'(summarize|abstract|summary)', user_question, re.I):
        return "summarize"

    return "search_db"

st.title("Smart Agent with Actions")

user_question = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_question:
        action = agent_decision(user_question)   
        if action == "greeting":
            st.write("Hello! How can I assist you today?")
        elif action == "summarize":
            text_to_summarize = st.text_area("Please enter the text you'd like summarized:")
            if text_to_summarize:
                summary = summarize_text(text_to_summarize)
                st.write("Here is the summary:")
                st.write(summary)
            else:
                st.write("Please provide text for summarization.")
        else:
            st.write("Processing your question with VectorDB search...")
            answer = user_input(user_question)
            st.write(answer['output_text'])
    else:
        st.write("Please enter a question.")
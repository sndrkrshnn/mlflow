###### IMPORTS ######
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

def data_preprocess():
    loader = TextLoader('./hp.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    out = FAISS.from_documents(docs, embedding=embeddings)
    out.save_local('local_db')
    return out 


db = data_preprocess()
query = input("Ask any question about Harry Potter and the Sorcerer's stone..")
llm=ChatOpenAI(model='gpt-4o',temperature=0.2,openai_api_key=API_KEY)
db_ret = db.as_retriever()
retriever=MultiQueryRetriever.from_llm(
    retriever=db_ret,
    llm=llm
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)
response = qa_chain.invoke("""
                           You are an expert data analyst.
                           Be accurate and detailed with your response.
                           If the content of the query is not within the boundary of the provided document, say you don't know.
                           Do not provide general knowledge answers.
                           Query:
                           """+query)
print(response['result'])
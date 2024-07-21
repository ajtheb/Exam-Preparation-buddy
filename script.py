from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
import os
from langchain_community.embeddings import HuggingFaceEmbeddings


class ExamPreparator:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="gemma:2b")
        self.prompt =  PromptTemplate.from_template(
                """
                <s> [INST] You are an expert in teaching.You will answer my queries in simple language using the context provided.  [/INST] </s> 
                [INST] Question: {question} 
                Context: {context} 
                Answer: [/INST]
                """
            )
        self.text_splitter = CharacterTextSplitter(
                                                    chunk_size= 800,
                                                    chunk_overlap = 100,
                                                    separator='', strip_whitespace=False
                                                )
        
        model_name = "sentence-transformers/all-mpnet-base-v2"
        # model_name = '/kaggle/input/stance-detect/transformers/1/1'
        model_kwargs = {"device": "cuda"}

        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
                
        
      

    def ingest(self,file_path):
        
        documents = []
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
        text_splitter = CharacterTextSplitter(
            chunk_size= 800,
            chunk_overlap = 100,
            separator='', strip_whitespace=False
        )
        all_splits = text_splitter.split_documents(documents)
        vectordb = Chroma.from_documents(documents=all_splits, embedding=self.embeddings, persist_directory="chroma_db")
        self.retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vectordb = None
        self.retriever = None
        self.chain = None

if __name__ == '__main__':
     d = ExamPreparator()
     d.ingest()
     answer = d.ask('What is cloud Native?')
     print(answer)
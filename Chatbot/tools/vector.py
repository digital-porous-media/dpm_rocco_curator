import streamlit as st
from llm import llm, embeddings
from graph import graph
from langchain_neo4j import Neo4jVector
from langchain.embeddings.base import Embeddings

# Create the Neo4jVector
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              # (1)
    graph=graph,                             # (2)
    index_name="datasetDescription",                 # (3)
    node_label="Dataset",                      # (4)
    text_node_property="description",               # (5)
    embedding_node_property="descriptionEmbedding", # (6)
    retrieval_query="""
RETURN
    node.description AS text,
    score,
    {
        title: node.title,
        sampleTitles: [(sample)-[:PART_OF]->(node) | sample.title ],
        datasetNumber: node.datasetNumber,
        doi: node.doi
    } AS metadata
"""
)
# Create the retriever
retriever = neo4jvector.as_retriever()
# Create the prompt
from langchain_core.prompts import ChatPromptTemplate

instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)
# Create the chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

question_answer_chain = create_stuff_documents_chain(llm, prompt)
description_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)
# Create a function to call the chain
def get_description(input):
    return description_retriever.invoke({"input": input})

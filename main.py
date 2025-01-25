import streamlit as st
from langchain_component import create_vectordb, get_qa_chain

st.title("Phone Recommender")

btn = st.button("create/update database")

if btn:
    create_vectordb()

question = st.text_input("Please give your desired specifications or details:")
if question:
    chain = get_qa_chain()
    response  = chain(question)

    st.header("Recommendation:")
    st.write(response["result"])
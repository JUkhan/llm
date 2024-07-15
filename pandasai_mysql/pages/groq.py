import os
from langchain.chains import LLMChain
from langchain_core.prompts import (
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import streamlit as st




st.title('Groq Chat App')

st.sidebar.title('Change memory length')
conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)
memory = ConversationBufferMemory(k=conversational_memory_length, memory_key='chat_history', return_messages=True)
system_prompt = 'You are a friendly conversational chatbot'

st.write(os.environ['GROQ_API_KEY'])
groq_chat = ChatGroq(
  model='mixtral-8x7b-32768',
  api_key=os.environ['GROQ_API_KEY']
)

user_question = st.text_area('Ask a question...')


if st.button('Generate'):
  if user_question:
    with st.spinner("Generating response..."):
      prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate.from_template('{human_input}')
      ])
      conversation = LLMChain(
        llm = groq_chat,
        prompt = prompt,
        verbose = False,
        memory = memory
      )
      response = conversation.predict(human_input = user_question)
      
      st.write('Chatbot', response)

from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from langchain_groq.chat_models import ChatGroq
import os
from pandasai import SmartDataframe

load_dotenv()

llm = ChatGroq(
  model='mixtral-8x7b-32768',
  api_key=os.environ['GROQ_API_KEY']
)

st.title('Data visualize PandasAI with groq')

file_upload= st.file_uploader('Upload file', type='csv')

if file_upload is not None:
  data=pd.read_csv(file_upload)
  st.write(data.head(3))

  df=SmartDataframe(data, config={'llm':llm})
  prompt = st.text_input('Enter your prompt:')
  if st.button('Generate:'):
    if prompt:
      with st.spinner('Generating response'):
        response = df.chat(prompt)
        if isinstance(response, str):
          st.write(response)
        else:
          st.json(response)
          st.write(response.choices[0].delta.content)

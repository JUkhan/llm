from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe

model=LocalLLM(
  api_base='http://localhost:11434/v1',
  model='llama3'
)

st.title('Data analysis with PandasAi')

upload_file = st.sidebar.file_uploader('Upload a csv file', type=['csv'])

if upload_file is not None:
   data = pd.read_csv(upload_file)
   st.write(data.head(3))

   df = SmartDataframe(data, config={'llm': model})

   prompt=st.text_area('Enter your prompt:')

   if(st.button('Generate')):
      if prompt:
         with st.spinner('Generating response...'):
            st.write(df.chat(prompt))
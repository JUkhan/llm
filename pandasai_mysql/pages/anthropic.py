from dotenv import load_dotenv
import streamlit as st
import seaborn as sns
from langchain_anthropic import ChatAnthropic
from pandasai import SmartDataframe

load_dotenv()

st.title('Data visualization by PandasAI with seaborn')
data = sns.load_dataset('penguins')
st.write(data.head(3))

model = ChatAnthropic(model='claude-3-haiku-20240307')
df=SmartDataframe(data, config={'llm':model})

st.write('''
# Sample prompt
- How many :rainbow[rows] and :rainbow[columns] are in this dataset?
- Tabulate summary statistics of data
- Tabulate how many missing value there are for each column.
- Draw the bar chart of penguin species.
- Draw a bar chart of the island, using different color for each bar.
- Draw a pie chart of the sex.
- Draw a sector plot of bill and depth columns, using a different color for each species.
- Draw a histogram of the flipper length with a kernel density estimate.
- Plot a heatmap chart of numerical variables.
- Create a box plot of bill length by island,                         
''')

prompt=st.text_input('Enter your prompt:')

if st.button('Generate'):
  if prompt:
    with st.spinner('Generating response...'):
      st.write(df.chat(prompt))

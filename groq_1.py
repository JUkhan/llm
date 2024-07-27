from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser



load_dotenv()

output_parser = StrOutputParser()

chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat | output_parser
#response = chain.invoke({"text": "Explain the importance of low latency LLMs."})
#print(response.content)

for res in chain.stream({'text':'Write a limerick about the Moon'}):
    print(res, end="", flush=True)
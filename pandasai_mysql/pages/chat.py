import streamlit as st
from typing import Generator
from groq import Groq

st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Streamed response emulator
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    #st.json(chat_completion)
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
      chat_completion = client.chat.completions.create(
              model='mixtral-8x7b-32768',
              messages=[
                  {
                      "role": m["role"],
                      "content": m["content"]
                  }
                  for m in st.session_state.messages
              ],
              max_tokens=32768,
              stream=True
          )
      with st.chat_message("assistant"):
        chat_responses_generator = generate_chat_responses(chat_completion)
        full_response = st.write_stream(chat_responses_generator)
    
    except Exception as e:
      st.error(e, icon="🚨")
    # Add assistant response to chat history
    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})
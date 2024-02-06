import streamlit as st
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


prompt = PromptTemplate(
    input_variables= ["chat_history", "question"],
    template="""You are a funny and witty AI, you are talking to a human
      answer him/her with some humor and wit
      
      chat_history: {chat_history}

      Human: {question}
      
      AI:"""
)

llm = Ollama(base_url="http://localhost:11434",model="llama2")
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5)
llm_chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)

st.set_page_config(
    page_title="Ollama Chat",
    page_icon="🦙",
    layout="wide"
)

st.title("Ollama Chat")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant","content": "Yo! start talking"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user","content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Waaiiitttt...."):
            ai_response= llm_chain.predict(question=user_prompt)
            st.write(ai_response)
    new_ai_message = {"role":"assistant", "content":ai_response}
    st.session_state.messages.append(new_ai_message)
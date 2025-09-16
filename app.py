import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper , WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun ,DuckDuckGoSearchResults
from langchain.agents import initialize_agent , AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()


## Arxiv and wikipedia and search tools

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=3,doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=3,doc_content_chars_max=300)  ## It will return wikipedia page sumaries top 3 results
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

search = DuckDuckGoSearchResults(name="Search")



## Streamlit app

st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar setting 

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq_api_key:",type="password")


## Session state

if "messages" not in st.session_state:
    st.session_state["messages"]=[]

    st.session_state["messages"].append({"role": "user", "content": "Hello!"})


for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


## prompt

if prompt := st.chat_input(placeholder="What is Machine Learning"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)


    llm = ChatGroq(groq_api_key = api_key,model_name="llama-3.1-8b-instant",streaming=True)
    tools = [search,arxiv,wiki]

    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)

    with st.chat_message("Assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)

        user_input = st.session_state.messages[-1]["content"]

        response = search_agent.run(user_input, callbacks=[st_cb])

        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)













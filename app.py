import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ----------------- TOOL SETUP -----------------
# Arxiv tool
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# Wikipedia tool
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

# Web search tool
search = DuckDuckGoSearchResults(name="Search")

# ----------------- STREAMLIT UI -----------------
st.title("🔎 LangChain - Chat with Search")
st.write("Chat with multiple tools: Wikipedia, Arxiv, and Web Search 🚀")

# Sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq_api_key:", type="password")

# Session state for storing messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["messages"].append({"role": "assistant", "content": "Hello! Ask me anything 😊"})

# Display previous chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ----------------- MAIN CHAT -----------------
if prompt := st.chat_input(placeholder="Ask me anything, e.g. 'What is Machine Learning?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        streaming=True
    )

    # Tools available
    tools = [search, arxiv, wiki]

    # Memory to keep track of conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # ----------------- CUSTOM INSTRUCTION -----------------
    prefix = """You are a helpful research assistant.
    - If the user asks about a research paper, author, or publication, ALWAYS use the Arxiv tool to fetch results.
    - After fetching results from Arxiv, combine them with your own knowledge to give a richer, more complete explanation.
    - For general knowledge or history, use Wikipedia.
    - For current events or web results, use Search.
    - Always provide a clear, detailed final answer that blends tool outputs with your reasoning.
    """

    # Create the agent
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        max_iterations=3,   # Prevent infinite loops
        memory=memory,
        verbose=True,
        agent_kwargs={"prefix": prefix}
    )

    # ----------------- RESPONSE HANDLING -----------------
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        response = search_agent.run(prompt, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

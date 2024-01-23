import streamlit as st
from streaming import StreamHandler

# from urllib.parse import quote_plus
from langchain.utilities import SQLDatabase

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains import create_sql_query_chain

st.title("Text2SQL Chatbot")
st.subheader("Allows users to interact with their DB powered by LLM")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_resource
def get_llm():
    # st_cb = StreamHandler(st.empty())
    # callback_manager = CallbackManager([st_cb])

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        model_path=r".\models\llama-2-7b.Q4_K_M.gguf",
        temperature=0.0,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
        n_ctx=4096,
        n_gpu_layers=33,
        n_batch=512,
    )

    return llm


@st.cache_resource
def connect_db():
    # Create DB connection string for mssql

    # conn_str = r'mssql+pymssql://{}:{}@{}:{}/{}'
    # username = "sa"
    # password = quote_plus("")
    # server_ip = ""
    # port = ""
    # DatabaseName = ""

    # conn_string = conn_str.format(username, password, server_ip, port, DatabaseName)

    conn_string = "sqlite:///./db/Chinook.db"

    # Instantiate the Database
    db = SQLDatabase.from_uri(conn_string, sample_rows_in_table_info=0)
    return db


@st.cache_resource
def get_sql_chain():
    sql_query_chain = create_sql_query_chain(get_llm(), connect_db())
    return sql_query_chain


if prompt := st.chat_input("How can I help you?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if (
    len(st.session_state["messages"]) > 0
    and st.session_state["messages"][-1]["role"] != "assistant"
):
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            sql_query_chain = get_sql_chain()
            response = sql_query_chain.invoke(
                {"question": st.session_state["messages"][-1]["content"]}
            )
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

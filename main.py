import streamlit as st

st.set_page_config(
    page_title="AI Nutrition & Maternal Care",
    page_icon="🧑‍🍼",
    layout="wide",
)

from utils.session_init import init_session

from ui.styles import inject_styles
from ui.header import render_header
from ui.sidebar import render_sidebar
from ui.chat_window import (
    restore_chat_state,
    render_chat,
)
from ui.input_bar import render_input

from services.chat_service import (
    load_messages,
    conversation_exists,
)

from core.llm import get_llms
from core.rag import get_rag_chain
from services.chat_flow import handle_prompt

init_session()
inject_styles()
render_header()
render_sidebar()

restore_chat_state(
    load_messages,
    conversation_exists,
)

render_chat()

llms = get_llms()
rag_chain = get_rag_chain(llms)
translator = llms.get("llm_translator")

prompt, uploaded_image = render_input()

if prompt:

    handle_prompt(
        prompt,
        uploaded_image,
        llms,
        rag_chain,
        translator,
    )
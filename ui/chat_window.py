import streamlit as st

def restore_chat_state(load_messages, conversation_exists):

    params = st.query_params
    active_id = params.get("chat")

    if active_id and conversation_exists(active_id):
        st.session_state.conversation_id = active_id
    elif "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    cid = st.session_state.conversation_id

    if st.session_state.get("last_loaded_chat") != cid:

        msgs = load_messages(cid) if cid else []

        st.session_state.messages = [
            {
                "role": m.role,
                "content": m.content,
                "audio_path": m.audio_path,
            }
            for m in msgs
        ]

        st.session_state.last_loaded_chat = cid


def render_chat():

    cid = st.session_state.get("conversation_id")

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    for m in st.session_state.get("messages", []):

        with st.chat_message(m["role"]):

            st.markdown(m["content"])

            if m["role"] == "assistant" and m.get("audio_path"):
                st.audio(m["audio_path"])

    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()
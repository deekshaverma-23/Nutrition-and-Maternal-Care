import streamlit as st

from services.chat_service import (
    get_all_conversations,
    delete_conversation,
    get_latest_conversation_id,
)

def render_sidebar():

    with st.sidebar:
        st.markdown(
            """
        <h2>🧑‍🍼 AI Nutrition</h2>
        <p style="opacity:0.6;margin-top:-10px;">Personalized care assistant</p>
        """,
            unsafe_allow_html=True,
        )

        if st.button("👤 Profile / Settings", use_container_width=True):

            chat_id = st.session_state.get("conversation_id")

            if chat_id:
                st.query_params["chat"] = chat_id
            elif "chat" in st.query_params:
                del st.query_params["chat"]

            st.switch_page("pages/profile.py")


        st.divider()

        st.markdown("### 💬 Chats")

        if st.button("➕ New Chat", use_container_width=True):

            st.session_state.conversation_id = None

            if "chat" in st.query_params:
                del st.query_params["chat"]


        convos = get_all_conversations()
        active_id = st.session_state.get("conversation_id")

        with st.container(height=480):

            for c in convos:

                label = (
                    c.title
                    if c.title and c.title != "New Chat"
                    else f"{c.created_at:%b %d %H:%M}"
                )

                is_active = c.id == active_id

                row = st.columns([8, 1])
                with row[0]:

                    if is_active:
                        st.markdown(
                            f"""
                            <div style="
                            padding:8px;
                            border-radius:8px;
                            background:#1f2933;
                            font-weight:600;
                            ">
                            ▶ {label}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        if st.button(
                            label,
                            key=f"open-{c.id}",
                            use_container_width=True,
                        ):
                            st.session_state.conversation_id = c.id
                            st.query_params["chat"] = c.id

                with row[1]:

                    if st.button("🗑", key=f"del-{c.id}"):

                        delete_conversation(c.id)
                        if c.id == active_id:

                            next_id = get_latest_conversation_id()

                            st.session_state.conversation_id = next_id

                            if next_id:
                                st.query_params["chat"] = next_id
                            elif "chat" in st.query_params:
                                del st.query_params["chat"]

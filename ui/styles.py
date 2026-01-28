import streamlit as st


def inject_styles():

    st.markdown(
        """
<style>
[data-testid="stSidebarNav"] { display: none; }

.chat-container {
    max-width: 900px;
    margin: auto;
}
</style>
""",
        unsafe_allow_html=True,
    )

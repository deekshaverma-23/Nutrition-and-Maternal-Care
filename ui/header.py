import streamlit as st

def render_header():

    col1, col2 = st.columns([1, 4])

    with col1:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/2966/2966480.png",
            width=80,
        )

    with col2:
        st.title("AI Nutrition & Maternal Care Assistant")
        st.caption("Smart guidance for mothers & families 🤍")

    st.divider()

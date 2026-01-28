import streamlit as st


def render_input():

    uploaded = st.file_uploader(
        "📷 Upload an image (optional)",
        type=["png", "jpg", "jpeg"],
        key=f"image_upload_{st.session_state.uploader_key}",
    )

    prompt = st.chat_input(
        "Ask about nutrition, pregnancy, baby care…"
    )

    return prompt, uploaded

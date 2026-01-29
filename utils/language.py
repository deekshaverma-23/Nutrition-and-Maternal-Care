import streamlit as st

LANG_CODE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Odia": "or",
}


def get_user_lang(session_state):

    return LANG_CODE_MAP.get(
        session_state.get("language", "English"),
        "en",
    )
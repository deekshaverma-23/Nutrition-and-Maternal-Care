import streamlit as st
from db import init_db
from utils.profile_state import load_profile


def init_session():

    init_db()

    if "profile_loaded" not in st.session_state:
        load_profile(force=True)
        st.session_state.profile_loaded = True

    if "audio_map" not in st.session_state:
        st.session_state.audio_map = {}

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
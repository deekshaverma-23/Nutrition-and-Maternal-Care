import streamlit as st
from utils.profile_state import load_profile, save_profile


def save_profile_and_refresh():
    save_profile()
    st.session_state["profile_loaded"] = False

load_profile(force=True)

st.set_page_config(
    page_title="Profile Settings",
    page_icon="👤",
    layout="wide",
)

st.markdown(
    """
<style>
[data-testid="stSidebarNav"] {display:none;}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:

    if st.button("⬅ Back to Chat", use_container_width=True):
        st.switch_page("main.py")

def on_user_type_change():

    t = st.session_state.user_type

    if t == "Pregnant Woman":
        st.session_state.age_months = ""
        st.session_state.age_years = None

    elif t == "Infant (0-12 months)":
        st.session_state.stage = ""
        st.session_state.diet = ""
        st.session_state.age_years = None

    else:  # Child
        st.session_state.stage = ""
        st.session_state.diet = ""
        st.session_state.age_months = ""

    save_profile_and_refresh()

st.title("👤 Profile & Health Information")
st.caption("Used to personalize nutrition guidance. Not medical advice.")

with st.container(border=True):

    st.selectbox(
        "Who is this for?",
        [
            "Pregnant Woman",
            "Infant (0-12 months)",
            "Child (1-5 years)",
        ],
        key="user_type",
        on_change=on_user_type_change,
    )

    if st.session_state.user_type == "Pregnant Woman":

        st.selectbox(
            "Stage",
            [
                "1st Trimester",
                "2nd Trimester",
                "3rd Trimester",
                "Postpartum",
            ],
            key="stage",
            on_change=save_profile_and_refresh,
        )

        st.selectbox(
            "Diet",
            ["Vegetarian", "Non-Vegetarian", "Vegan"],
            key="diet",
            on_change=save_profile_and_refresh,
        )
    elif st.session_state.user_type == "Infant (0-12 months)":

        st.selectbox(
            "Age (months)",
            ["0-6 months", "7-12 months"],
            key="age_months",
            on_change=save_profile_and_refresh,
        )
    else:
        if st.session_state.get("age_years") is None:
            st.session_state["age_years"] = 2

        st.slider(
            "Age",
            min_value=1,
            max_value=5,
            step=1,
            key="age_years",
            on_change=save_profile_and_refresh,
        )

    st.text_input(
        "Conditions",
        key="conditions",
        on_change=save_profile_and_refresh,
    )

    st.selectbox(
        "Preferred Language",
        [
            "English",
            "Hindi",
            "Bengali",
            "Tamil",
            "Telugu",
            "Marathi",
            "Gujarati",
            "Kannada",
            "Malayalam",
            "Punjabi",
            "Odia",
        ],
        key="language",
        on_change=save_profile_and_refresh,
    )


st.success("Profile saved automatically ✔")

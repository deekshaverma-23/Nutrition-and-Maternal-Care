from db import SessionLocal, UserProfile
import streamlit as st

PROFILE_ID = "main"

DEFAULT_PROFILE = {
    "user_type": "Pregnant Woman",
    "stage": "1st Trimester",
    "diet": "Vegetarian",
    "age_months": "0-6 months",
    "age_years": "2",
    "child_status": "Healthy",
    "conditions": "",
    "language": "en",
}

def load_profile(force=False):

    db = SessionLocal()

    profile = (
        db.query(UserProfile)
        .filter(UserProfile.id == PROFILE_ID)
        .first()
    )

    if not profile:
        profile = UserProfile(
            id=PROFILE_ID,
            **DEFAULT_PROFILE,
        )
        db.add(profile)
        db.commit()

    for k in DEFAULT_PROFILE:

        val = getattr(profile, k)
        if k == "age_years":
            try:
                val = int(val)
            except:
                val = DEFAULT_PROFILE["age_years"]

        if k in ["stage", "diet", "user_type", "age_months", "conditions"]:
            val = val or DEFAULT_PROFILE[k]

        st.session_state[k] = val
    db.close()

def save_profile():

    db = SessionLocal()

    profile = (
        db.query(UserProfile)
        .filter(UserProfile.id == PROFILE_ID)
        .first()
    )

    if not profile:
        profile = UserProfile(id=PROFILE_ID)

    for k in DEFAULT_PROFILE:

        v = st.session_state.get(k)
        if isinstance(v, tuple):
            v = v[0] if v else None

        if k == "age_years":
            if v in ("", None):
                v = None
            else:
                v = int(v)


        setattr(profile, k, str(v) if v is not None else "")



    db.add(profile)
    db.commit()
    db.close()

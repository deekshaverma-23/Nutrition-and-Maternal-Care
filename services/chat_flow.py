import streamlit as st
from PIL import Image

from utils.language import get_user_lang
from utils.audio import text_to_speech_to_file

from services.chat_service import (
    bootstrap_conversation,
    save_message,
    get_conversation,
    update_conversation_title,
    generate_title,
)

from core.vision import get_image_response

VISION_TRIGGER_WORDS = [
    "image",
    "photo",
    "picture",
    "this",
    "food",
    "eat",
    "can i",
    "is this",
]

def handle_prompt(
    prompt,
    uploaded_image,
    llms,
    rag_chain,
    translator,
):

    user_lang = get_user_lang(st.session_state)
    if not st.session_state.conversation_id:

        cid = bootstrap_conversation()
        st.session_state.conversation_id = cid
        st.query_params["chat"] = cid

    save_message(st.session_state.conversation_id, "user", prompt)

    with st.chat_message("user"):

        if uploaded_image:
            st.image(uploaded_image, width=300)

        st.markdown(prompt)

    with st.spinner("🧠 Thinking..."):

        use_image = (
            uploaded_image
            and any(w in prompt.lower()
                    for w in VISION_TRIGGER_WORDS)
        )

        if use_image:

            final_response = get_image_response(
                Image.open(uploaded_image),
                prompt,
                llms,
            )

        else:

            augmented = (
                f"Profile:\nType: {st.session_state.get('user_type')}"
                f"\nQuestion: {prompt}"
            )

            result = rag_chain.invoke({"input": augmented})

            answer_en = result["answer"]

            if user_lang != "en" and translator:

                target = st.session_state.get("language", "English")

                translated = translator.invoke(
                    f"""
                You are a professional native translator.

                Task:
                Translate the text below into {target}.

                Rules:
                - Return ONLY the translated text.
                - Do NOT include explanations, titles, notes, or prefixes.
                - Do NOT mention that this is a translation.
                - Use natural, fluent, conversational language.
                - Prefer Indian cultural phrasing where appropriate.
                - Keep medical meaning accurate.
                - Keep tone calm and helpful.

                Text:
                {answer_en}
                """
                )

                final_response = str(translated).strip()

            else:
                final_response = answer_en

    msg_count = len(st.session_state.get("messages", []))

    audio_key = (
        f"{st.session_state.conversation_id}_"
        f"{msg_count}"
    )


    with st.chat_message("assistant"):

        st.markdown(final_response)

        audio_path = text_to_speech_to_file(
            final_response,
            user_lang,
            audio_key,
        )

        if audio_path:
            st.audio(audio_path)

    save_message(
        st.session_state.conversation_id,
        "assistant",
        final_response,
        audio_path=audio_path,
    )

    convo = get_conversation(st.session_state.conversation_id)

    if convo and convo.title == "New Chat":

        MAIN_LLM = (
            llms.get("llm_main")
            or llms.get("llm")
            or llms.get("chat")
            or llms.get("main")
            or next(iter(llms.values()))
        )

        title = generate_title(MAIN_LLM, prompt)

        update_conversation_title(
            st.session_state.conversation_id,
            title,
        )

        st.rerun()

import streamlit as st
from dotenv import load_dotenv
import asyncio
from langdetect import detect
from gtts import gTTS
from io import BytesIO
from PIL import Image
import base64

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate

asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv()


@st.cache_resource
def get_llms():
    """Returns a dictionary of initialized LLMs."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    llm_vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)
    llm_translator = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    return {"llm": llm, "llm_vision": llm_vision, "llm_translator": llm_translator}

def image_to_base64_str(pil_image):
    """Converts a Pillow image to a base64 data URL string."""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def translate_text(text, target_language, llms):
    """Uses an LLM to translate text."""
    prompt = f"Translate the following text to {target_language}: {text}"
    response = llms["llm_translator"].invoke(prompt)
    return response.content

def detect_language(text):
    """Detects the language of the given text."""
    try:
        return detect(text)
    except:
        return "en"

def text_to_speech(text, lang):
    """Converts text to speech and returns audio data."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"Error in TTS: {e}")
        return None


DB_PATH = "db/"

@st.cache_resource
def get_rag_chain(_llms):
    """Creates and returns a personalized conversational RAG chain."""
    system_template = """
    You are a helpful AI assistant for nutrition and maternal care, specifically for people in India.
    Your answers should be simple, practical, and use locally understandable terms. 
    **Crucially, you must respond in the same language as the user's question.** For example, if the question is in Hindi, respond in Hindi. If it is in English, respond in English.
    Use the retrieved context and the user's personal profile to answer the question.
    Always tailor your response to the user's specific profile, whether they are a pregnant woman, an infant, or a child. For infants, focus on breastfeeding and appropriate complementary foods after 6 months. For malnourished children, suggest energy-dense and protein-rich local foods.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.

    CONTEXT: {context}
    
    USER'S QUESTION: {question}
    
    Answer in a clear, helpful, and personalized way:
    """
    
    QA_PROMPT = PromptTemplate.from_template(system_template)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=_llms["llm"],
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT} 
    )
    return conversational_chain

def get_image_response(image_obj, prompt, llms):
    """Gets a response from the vision model for an image and prompt."""
    image_url = image_to_base64_str(image_obj)
    human_message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": image_url},
        ]
    )
    response = llms["llm_vision"].invoke([human_message])
    return response.content


st.set_page_config(page_title="AI Nutrition & Maternal Care Assistant", page_icon="🧑‍🍼")
st.title("AI Nutrition & Maternal Care Assistant")

with st.sidebar:
    st.header("👤 Your Profile")
    st.warning("This AI is for informational purposes, not medical advice. Always consult a doctor.")
    
    if "user_type" not in st.session_state:
        st.session_state.user_type = "Pregnant/Lactating Woman"
    
    st.session_state.user_type = st.selectbox(
        "Who is this advice for?",
        ["Pregnant/Lactating Woman", "Infant (0-12 months)", "Child (1-5 years)"],
    )

    if st.session_state.user_type == "Pregnant/Lactating Woman":
        st.session_state.stage = st.selectbox("Stage:", ["1st Trimester", "2nd Trimester", "3rd Trimester", "Postpartum/Lactating"])
        st.session_state.diet = st.selectbox("Dietary Preference:", ["Vegetarian", "Non-Vegetarian", "Vegan"])
    
    elif st.session_state.user_type == "Infant (0-12 months)":
        st.session_state.age_months = st.selectbox("Infant's Age:", ["0-6 months", "7-12 months"])
    
    elif st.session_state.user_type == "Child (1-5 years)":
        st.session_state.age_years = st.slider("Child's Age (years):", 1, 5, 2)
        st.session_state.child_status = st.selectbox("Nutritional Status:", ["Healthy", "Underweight", "Stunted"])

    st.session_state.conditions = st.text_input("Allergies or Health Conditions:", "None")

llms = get_llms()

if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.chat_message("assistant"):
        st.markdown("Hello! Please set the user profile in the sidebar, then ask me a question.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message:
            st.image(message["image"], width=200)
        st.markdown(message["content"])

rag_chain = get_rag_chain(llms)

uploaded_image = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])
image_data = None
if uploaded_image:
    image_data = Image.open(uploaded_image)

if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        if image_data:
            st.image(image_data, width=200)
            st.session_state.messages.append({"role": "user", "content": prompt, "image": image_data})
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        final_response = ""
        if image_data:
            final_response = get_image_response(image_data, prompt, llms)
        else:
            user_profile = f"User Profile: [Category: {st.session_state.user_type}, "
            if st.session_state.user_type == "Pregnant/Lactating Woman":
                user_profile += f"Stage: {st.session_state.stage}, Diet: {st.session_state.diet}, "
            elif st.session_state.user_type == "Infant (0-12 months)":
                user_profile += f"Age: {st.session_state.age_months}, "
            elif st.session_state.user_type == "Child (1-5 years)":
                user_profile += f"Age: {st.session_state.age_years} years, Status: {st.session_state.child_status}, "
            user_profile += f"Conditions: {st.session_state.conditions}]"
            
            augmented_prompt = f"{user_profile}\n\nQuestion: {prompt}"
            
            original_lang = detect_language(prompt)
            prompt_in_english = translate_text(augmented_prompt, "English", llms) if original_lang != "en" else augmented_prompt
            
            result = rag_chain.invoke({"question": prompt_in_english})
            response_in_english = result["answer"]
            
            if original_lang != "en":
                language_map = {"hi": "Hindi", "bn": "Bengali", "te": "Telugu", "mr": "Marathi", "ta": "Tamil"}
                target_language_name = language_map.get(original_lang, "the original language")
                final_response = translate_text(response_in_english, target_language_name, llms)
            else:
                final_response = response_in_english
        
        with st.chat_message("assistant"):
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            
            response_lang = detect_language(final_response)
            audio_data = text_to_speech(final_response, response_lang)
            if audio_data:
                st.audio(audio_data, format='audio/mp3')
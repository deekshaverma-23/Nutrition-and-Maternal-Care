## Personalized AI Nutrition & Maternal Care Assistant 

An AI-powered solution designed to bridge the healthcare information gap for vulnerable communities in rural India.

This project is a helpful chatbot engineered to provide personalized nutrition and health guidance. It specifically addresses the unique needs of pregnant women, infants, and malnourished children within the cultural and socio-economic landscape of rural Uttar Pradesh. It's more than a Q&A bot; it's an accessible, localized, and empathetic health companion.

---

##  Core Features & Innovations
This assistant is built on four key pillars that make it uniquely effective for its target audience:

### 1. Hyper-Personalized Profiles
The AI's intelligence goes beyond generic advice. It creates a dynamic profile for each user, allowing it to provide guidance that is precisely tailored to one of three distinct, vulnerable groups:
* *Pregnant & Lactating Women:* Advice changes based on the trimester or postpartum stage.
* *Infants (0-12 months):* Guidance focuses on breastfeeding and the safe introduction of complementary foods.
* *Children (1-5 years):* Recommendations are adjusted based on age and nutritional status (e.g., underweight, stunted).

### 2. Hyper-Local RAG System
The application's "brain" is a Retrieval-Augmented Generation (RAG) system built on a custom knowledge base. Instead of relying on generic web data, it retrieves information from documents curated for:
* *Indian Health Guidelines:* Sourced from official health ministries and organizations.
* *Local Food Ecosystem:* Considers foods that are affordable, available, and culturally accepted in Uttar Pradesh.
* *Socio-Economic Realities:* Understands budget constraints and local dietary habits.

### 3. Accessibility-First Interface
Designed for users with varying levels of literacy and language preferences, the interface breaks down barriers to information:
* *Seamless Multilingual Chat:* Automatically detects and communicates in multiple Indian languages (like Hindi) and English.
* *Text-to-Speech (TTS):* Every AI response can be read aloud with the click of a button, ensuring comprehension for all users.

### 4. Intuitive Multimodal Input
Understanding that a picture is worth a thousand words, the assistant can *see and analyze images*. Users can simply upload a photo of a meal or a local vegetable and ask questions like, "Is this healthy for my child?" This makes interaction natural and removes the need to describe unknown food items.

---

## System Architecture

The application employs a sophisticated, multi-stage pipeline to handle user queries:

1.  *Input:* The Streamlit UI captures user input, which can be text, an image, or both, along with profile data from the sidebar.
2.  *Triage:* The system first determines if the query involves an image.
3.  *Vision Path:* Image-based queries are routed directly to the *Google's gemini-1.5-flash* model for analysis.
4.  *RAG Path:* Text-based queries are augmented with the user's profile. The language is detected, and the query is translated to English for optimal performance.
5.  *Retrieval:* The query is used to retrieve the most relevant document chunks from the *ChromaDB* vector store.
6.  *Generation:* The retrieved context, the user's question, and the personalized prompt are sent to *Google's gemini-1.5-flash* to generate a nuanced, context-aware answer.
7.  *Output:* The generated response is translated back to the user's original language and delivered as both text and audible speech (TTS).

---

## Technology Showcase

* *LangChain:* The core framework used to orchestrate the complex RAG pipeline, manage conversational memory, and chain together different LLM calls.
* *Streamlit:* Enables the rapid development of a rich, interactive, and user-friendly web interface.
* *ChromaDB:* A high-performance vector database that allows for efficient local semantic search of the custom knowledge base.

---

## Getting Started

Follow these steps to set up and run the project locally.

1.  *Clone the Repository:* git clone <your-repository-url>
2.  *Set Up Environment:* Create and activate a Python virtual environment.
3.  *Install Dependencies:* pip install -r requirements.txt
4.  *Add API Key:* Create a .env file and add your GOOGLE_API_KEY.
5.  *Run the Application:* streamlit run main.py.

import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from groq import Groq
import os

# Hardcode your Groq API key here
API_KEY = "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"  # Replace with your actual API key

# Hardcode the FAQ file path
FAQ_FILE_PATH = "faq.xlsx"  # Make sure this file is in the same directory as your script

groq_client = Groq(api_key=API_KEY)

# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load and embed FAQ data
@st.cache_data(show_spinner=False)
def load_faq_embeddings(faq_path):
    df = pd.read_excel(faq_path, engine='openpyxl')
    df = df.dropna(subset=['Question', 'Answer'])
    df['embedding'] = df['Question'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

# Update embeddings after new info
def update_faq(df, new_q, new_a):
    new_embed = model.encode(new_q, convert_to_tensor=True)
    return pd.concat([df, pd.DataFrame([{
        "Question": new_q,
        "Answer": new_a,
        "embedding": new_embed
    }])], ignore_index=True)

# Retrieve top-k relevant questions
def retrieve_top_k(user_question, faq_df, k=5):
    question_embedding = model.encode(user_question, convert_to_tensor=True)
    faq_df['similarity'] = faq_df['embedding'].apply(lambda x: util.cos_sim(question_embedding, x).item())
    top_k = faq_df.sort_values(by='similarity', ascending=False).head(k)
    return top_k

# Generate AI response using Groq with selected model
def generate_ai_response(user_question, relevant_context, system_prompt, selected_model):
    if not groq_client:
        return "Groq client not initialized. Please check your API key."
    
    # Prepare context from retrieved FAQs
    context = "\n\n".join([f"Q: {row['Question']}\nA: {row['Answer']}" 
                          for _, row in relevant_context.iterrows()])
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": f"""You are a helpful assistant Priya, working at Narayan Seva Sansthan (NSS). Based on the following FAQ context, please answer the user's question.

Context:
{context}

User Question: {user_question}

Please provide a helpful and accurate response based on the context provided. If the context doesn't contain enough information to answer the question, please say so."""
        }
    ]
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=selected_model,  # Use the selected model
            temperature=0.7,
            max_tokens=150
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating AI response: {str(e)}"

# -------------- Streamlit UI ----------------
st.title("📚 Semantic FAQ Retriever with AI Assistant")

# System prompt and model configuration
with st.expander("🤖 AI Assistant Configuration"):
    system_prompt = st.text_area(
        "System Prompt",
        value="""You are Priya, a helpful AI assistant working at Narayan Seva Sansthan (NSS). You answer questions based on provided FAQ context. 
You should:
- Introduce yourself as Priya from NSS when greeting users
- Provide accurate answers based on the given context
- Be concise but comprehensive
- If information is not available in the context, clearly state so and offer to help find more information
- Use a friendly and professional tone appropriate for NSS
- Format your responses clearly""",
        height=150,
        help="Define how the AI assistant should behave and respond to questions"
    )
    
    # Model selection that will be used for AI responses
    model_choice = st.selectbox(
        "Select Groq Model for AI Responses",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        index=0,
        help="Choose the Groq model for generating AI responses"
    )

# Load FAQ data from hardcoded path
if os.path.exists(FAQ_FILE_PATH):
    faq_df = load_faq_embeddings(FAQ_FILE_PATH)
    
    # Show Top-K slider
    k = st.slider("Select number of relevant answers to retrieve (Top K)", min_value=1, max_value=10, value=3)
    
    # Input question
    user_question = st.text_input("Ask a question:")
    
    # Add new knowledge section
    with st.expander("➕ Add new FAQ / info manually"):
        new_q = st.text_input("New question/information (e.g., What is NSS?)", key="new_q")
        new_a = st.text_input("New answer (e.g., NSS is Narayan Seva Sanstha)", key="new_a")
        if st.button("Add to knowledge base"):
            if new_q and new_a:
                faq_df = update_faq(faq_df, new_q, new_a)
                st.success("New info added successfully!")
            else:
                st.warning("Both fields are required.")
    
    # Show results when user asks a question
    if user_question:
        # Retrieve relevant FAQs using semantic search
        results = retrieve_top_k(user_question, faq_df, k)
        
        # Automatically generate AI response using the retrieved context
        with st.spinner("🔍 Finding relevant information and generating response..."):
            ai_response = generate_ai_response(
                user_question, 
                results, 
                system_prompt,
                model_choice  # Use the selected model
            )
        
        # Display AI response prominently
        st.subheader("🤖 Priya's Response:")
        st.markdown(f"**Model Used:** {model_choice}")
        st.markdown(ai_response)
        
        # Show the underlying semantic search results in an expander
        with st.expander("🔍 View Source Information (Semantic Search Results)"):
            st.caption(f"Retrieved {len(results)} most relevant FAQ entries:")
            for i, row in results.iterrows():
                st.markdown(f"**Q:** {row['Question']}")
                st.markdown(f"**A:** {row['Answer']}")
                st.caption(f"Similarity score: `{row['similarity']:.4f}`")
                st.markdown("---")

else:
    st.error(f"❌ FAQ file not found at: {FAQ_FILE_PATH}")
    st.info("Please make sure 'faq.xlsx' is in the same directory as your script.")
    
    # Instructions
    with st.expander("📋 Instructions"):
        st.markdown("""
        ### How to use:
        1. **Configure the AI assistant** - Set system prompt and select model
        2. **Upload FAQ Excel file** with columns: 'Question' and 'Answer'
        3. **Ask your question** - The system will automatically:
           - Find relevant FAQs using semantic search
           - Generate an AI response using the selected model
           - Show both the AI response and source information
        4. **Add new information** manually if needed
        
        ### Setup Requirements:
        - Replace the hardcoded API key with your actual Groq API key
        - Make sure 'faq.xlsx' file is in the same directory as your script
        - Excel file should have 'Question' and 'Answer' columns
        
        ### How it works:
        - **Step 1:** Semantic search finds the most relevant FAQs for your question
        - **Step 2:** Selected Groq model generates a response using those FAQs as context
        - **Step 3:** You get both the AI response and can view the source information
        """)
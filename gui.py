import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import google.generativeai as genai
import os
import PyPDF2
import pdfplumber
from PIL import Image
import pytesseract
import io
import re
from datetime import datetime
import json

# Hardcode your Gemini API key here
GEMINI_API_KEY = "my key"  # Replace with your actual Gemini API key

# Hardcode the FAQ file path
FAQ_FILE_PATH = "faq.xlsx"  # Make sure this file is in the same directory as your script

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_content' not in st.session_state:
    st.session_state.pdf_content = ""
if 'pdf_filename' not in st.session_state:
    st.session_state.pdf_filename = ""
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""

# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load and embed FAQ data
@st.cache_data(show_spinner=False)
def load_faq_embeddings(faq_path):
    if os.path.exists(faq_path):
        df = pd.read_excel(faq_path, engine='openpyxl')
        df = df.dropna(subset=['Question', 'Answer'])
        df['embedding'] = df['Question'].apply(lambda x: model.encode(x, convert_to_tensor=True))
        return df
    else:
        # Return empty dataframe if file doesn't exist
        return pd.DataFrame(columns=['Question', 'Answer', 'embedding'])

# PDF text extraction functions
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using multiple methods for better accuracy"""
    text = ""
    
    try:
        # Method 1: Try pdfplumber first (better for complex layouts)
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # If pdfplumber didn't extract much text, try PyPDF2
        if len(text.strip()) < 50:
            pdf_file.seek(0)  # Reset file pointer
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""
    
    return text.strip()

def parse_receipt_info(text):
    """Parse receipt information using regex patterns"""
    receipt_info = {}
    
    # Common receipt patterns
    patterns = {
        'date': [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(\d{1,2}\s+\w+\s+\d{2,4})',
            r'(Date:\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
        ],
        'total': [
            r'Total[:\s]*\$?(\d+\.?\d*)',
            r'Amount[:\s]*\$?(\d+\.?\d*)',
            r'TOTAL[:\s]*\$?(\d+\.?\d*)'
        ],
        'store': [
            r'^([A-Z\s&]+)(?=\n|\r)',  # First line often contains store name
            r'(.*?)(?=\n.*\d{3}[-.]?\d{3}[-.]?\d{4})'  # Text before phone number
        ],
        'items': [
            r'(\w+.*?)[\s]*\$(\d+\.?\d*)',  # Item name followed by price
        ]
    }
    
    for field, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                if field == 'items':
                    receipt_info[field] = matches[:10]  # Limit to first 10 items
                else:
                    receipt_info[field] = matches[0] if isinstance(matches[0], str) else matches[0][0] if matches[0] else ""
                break
    
    return receipt_info

# Update embeddings after new info
def update_faq(df, new_q, new_a):
    new_embed = model.encode(new_q, convert_to_tensor=True)
    new_row = pd.DataFrame([{
        "Question": new_q,
        "Answer": new_a,
        "embedding": new_embed
    }])
    return pd.concat([df, new_row], ignore_index=True)

# Retrieve top-k relevant questions
def retrieve_top_k(user_question, faq_df, k=5):
    if faq_df.empty:
        return pd.DataFrame()
    
    question_embedding = model.encode(user_question, convert_to_tensor=True)
    faq_df['similarity'] = faq_df['embedding'].apply(lambda x: util.cos_sim(question_embedding, x).item())
    top_k = faq_df.sort_values(by='similarity', ascending=False).head(k)
    return top_k

# Generate AI response with chat history and PDF context using Gemini
def generate_ai_response(user_question, relevant_context, system_prompt, selected_model, chat_history, pdf_content=""):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return "Gemini API key not configured. Please check your API key."
    
    # Prepare context from retrieved FAQs
    faq_context = ""
    if not relevant_context.empty:
        faq_context = "\n\n".join([f"Q: {row['Question']}\nA: {row['Answer']}" 
                              for _, row in relevant_context.iterrows()])
    
    # Prepare chat history context
    history_context = ""
    is_first_message = len(chat_history) == 0
    if chat_history:
        recent_history = chat_history[-3:]  # Last 3 exchanges for better context
        history_context = "\n".join([f"User: {h['user']}\nPriya: {h['assistant']}" 
                                   for h in recent_history if h.get('type') != 'system'])
    
    # Prepare PDF context
    pdf_context = ""
    if pdf_content:
        pdf_context = f"\n\nUploaded Document Content:\n{pdf_content[:2000]}..."  # Limit to 2000 chars
    
    # Create context-aware prompt
    conversation_context = ""
    if history_context:
        conversation_context = f"\n\nPREVIOUS CONVERSATION:\n{history_context}\n"
    
    # Combine all context
    full_prompt = f"""{system_prompt}

{conversation_context}FAQ KNOWLEDGE BASE:
{faq_context}

{pdf_context}

INSTRUCTIONS:
- This is {"the user's FIRST message" if is_first_message else "a FOLLOW-UP message in an ongoing conversation"}
- {"Introduce yourself with 'Jai Siyaram! I'm Priya, an AI assistant from Narayan Seva Sansthan'" if is_first_message else "Continue the conversation naturally without repeating introductions"}
- Be contextually aware of previous messages
- Provide helpful, accurate responses based on available information
- If referencing costs or donations, be clear about amounts and options

CURRENT USER QUESTION: {user_question}

Response:"""
    
    try:
        # Initialize Gemini model
        model_name = "gemini-1.5-flash" if selected_model == "gemini-flash" else "gemini-1.5-pro"
        gemini_model = genai.GenerativeModel(model_name)
        
        # Generate response
        response = gemini_model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=400,
            )
        )
        
        return response.text
    except Exception as e:
        return f"Error generating AI response: {str(e)}"

# -------------- Streamlit UI ----------------
st.title("📚 Enhanced FAQ System with PDF Processing & Chat History")

# Custom CSS for better styling
st.markdown("""
<style>
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        background-color: #f8fafc;
    }
    
    .user-message {
        border-left-color: #10b981;
        background-color: #f0fdf4;
    }
    
    .assistant-message {
        border-left-color: #6366f1;
        background-color: #faf5ff;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("🤖 Configuration")
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="""You are Priya, a helpful AI assistant working at Narayan Seva Sansthan (NSS). You answer questions based on provided FAQ context, uploaded documents, and chat history. 

IMPORTANT GUIDELINES:
- Only introduce yourself as "Jai Siyaram! I'm Priya, an AI assistant from Narayan Seva Sansthan" for the FIRST message or when greeting new users
- For follow-up questions in the same conversation, respond naturally without repeating the full introduction
- Maintain conversation flow by referencing previous messages when relevant
- Be contextually aware - if someone asks a follow-up question, acknowledge their previous question
- Provide clear, helpful answers based on the given context
- Use "Jai Siyaram!" as a greeting only when appropriate (first interaction or after long gaps)
- Be conversational and natural, not robotic or repetitive
- Reference uploaded documents when relevant
- Be concise but comprehensive""",
        height=250
    )
    
    # Model selection for Gemini
    model_choice = st.selectbox(
        "Select Gemini Model",
        ["gemini-flash", "gemini-pro"],
        index=0,
        help="Flash is faster, Pro is more capable"
    )
    
    # Top-K slider
    k = st.slider("Top K relevant answers", min_value=1, max_value=10, value=3)
    
    # Clear chat history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.processed_files = set()
        st.session_state.last_question = ""
        st.success("Chat history cleared!")

# Load FAQ data
faq_df = load_faq_embeddings(FAQ_FILE_PATH)

# Function to handle file upload and processing
def process_uploaded_file(uploaded_file):
    """Process uploaded PDF file and update session state"""
    if uploaded_file is not None:
        # Create a unique identifier for the file
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Check if this file has already been processed
        if file_id in st.session_state.processed_files:
            return False
        
        with st.spinner("📖 Processing PDF..."):
            # Extract text from PDF
            extracted_text = extract_text_from_pdf(uploaded_file)
            
            if extracted_text:
                st.session_state.pdf_content = extracted_text
                st.session_state.pdf_filename = uploaded_file.name
                st.session_state.processed_files.add(file_id)
                
                # Try to parse as receipt
                receipt_info = parse_receipt_info(extracted_text)
                
                # Add to chat history as a system message
                receipt_summary = f"📄 Uploaded document: {uploaded_file.name}"
                if receipt_info:
                    receipt_details = []
                    for key, value in receipt_info.items():
                        if value:
                            if isinstance(value, list):
                                receipt_details.append(f"{key.title()}: {', '.join(str(v) for v in value[:2])}")
                            else:
                                receipt_details.append(f"{key.title()}: {value}")
                    if receipt_details:
                        receipt_summary += f"\n🧾 Detected: {', '.join(receipt_details[:3])}"
                
                st.session_state.chat_history.append({
                    'user': f"[Uploaded: {uploaded_file.name}]",
                    'assistant': receipt_summary,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'system'
                })
                
                # Automatically add PDF content to knowledge base
                global faq_df
                pdf_summary = f"Document: {uploaded_file.name}"
                faq_df = update_faq(faq_df, pdf_summary, extracted_text[:500] + "...")
                
                return True
            else:
                st.error("❌ Could not extract text from the PDF. Please check if the file is valid.")
                return False
    return False

# Main interface tabs
tab1, tab2 = st.tabs(["💬 Chat", "➕ Manage Knowledge"])

with tab1:
    st.subheader("Chat with Priya")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                if chat.get('type') == 'system':
                    # System messages (file uploads)
                    st.info(chat['assistant'])
                else:
                    # Regular chat messages with better styling
                    col1, col2 = st.columns([1, 10])
                    with col2:
                        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {chat["user"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="chat-message assistant-message"><strong>Priya:</strong> {chat["assistant"]}</div>', unsafe_allow_html=True)
                        if i < len(st.session_state.chat_history) - 1:
                            st.markdown("<br>", unsafe_allow_html=True)
    
    # Chat input area
    st.markdown("---")
    
    # Simple file upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Create columns for input and send button
    col1, col2 = st.columns([9, 1])
    
    with col1:
        # Use a unique key that changes to clear input
        input_key = f"chat_input_{len(st.session_state.chat_history)}"
        user_question = st.text_input("Ask a question:", key=input_key, label_visibility="collapsed", placeholder="Type your message here...")
    
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Handle file upload
    if uploaded_file is not None:
        if process_uploaded_file(uploaded_file):
            st.success(f"✅ Processed: {uploaded_file.name}")
            st.rerun()
    
    # Handle text input - auto-clear after sending
    if user_question and send_button:
        with st.spinner("🤔 Priya is thinking..."):
            # Retrieve relevant FAQs
            results = retrieve_top_k(user_question, faq_df, k)
            
            # Generate AI response
            ai_response = generate_ai_response(
                user_question, 
                results, 
                system_prompt,
                model_choice,
                st.session_state.chat_history,
                st.session_state.pdf_content
            )
            
            # Add to chat history
            st.session_state.chat_history.append({
                'user': user_question,
                'assistant': ai_response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Clear the last question to avoid duplication
            st.session_state.last_question = ""
            
            # Rerun to refresh chat and clear input (new key will clear input)
            st.rerun()
    
    # Show source information for last response if available
    if st.session_state.chat_history and st.session_state.chat_history[-1].get('type') != 'system':
        last_user_question = st.session_state.chat_history[-1]['user']
        results = retrieve_top_k(last_user_question, faq_df, k)
        if not results.empty:
            with st.expander("📚 Source Information"):
                for i, row in results.iterrows():
                    st.markdown(f"**Q:** {row['Question']}")
                    st.markdown(f"**A:** {row['Answer']}")
                    st.caption(f"Similarity: {row['similarity']:.4f}")
                    st.markdown("---")
    
    # Current document status (if any)
    if st.session_state.pdf_filename:
        st.sidebar.success(f"📄 Document: {st.session_state.pdf_filename}")
        if st.sidebar.button("🗑️ Remove Document"):
            st.session_state.pdf_content = ""
            st.session_state.pdf_filename = ""
            st.rerun()

with tab2:
    st.subheader("➕ Manage Knowledge Base")
    
    # Show current FAQ count
    st.info(f"📊 Current knowledge base contains {len(faq_df)} entries")
    
    # Add new FAQ manually
    with st.form("add_faq"):
        st.write("**Add New FAQ Entry**")
        new_q = st.text_input("Question:")
        new_a = st.text_area("Answer:")
        submitted = st.form_submit_button("Add to Knowledge Base")
        
        if submitted and new_q and new_a:
            faq_df = update_faq(faq_df, new_q, new_a)
            st.success("✅ New FAQ added successfully!")
    
    # Show current FAQ entries
    if not faq_df.empty:
        with st.expander("📋 View All FAQ Entries"):
            for i, row in faq_df.iterrows():
                st.write(f"**Q:** {row['Question']}")
                st.write(f"**A:** {row['Answer']}")
                st.markdown("---")

# Instructions
with st.expander("📋 How to Use This System"):
    st.markdown("""
    ### Features:
    1. **Chat Interface** - Have continuous conversations with Priya
    2. **PDF Processing** - Upload receipts, invoices, or documents
    3. **Receipt Parsing** - Automatically extract key information from receipts
    4. **Knowledge Management** - Add custom FAQ entries
    5. **Chat History** - Maintains context across conversations
    6. **Semantic Search** - Find relevant information from all sources
    7. **Gemini AI** - Powered by Google's Gemini AI models
    
    ### How to Use:
    1. **Start Chatting** - Type your question in the input field
    2. **Upload Documents** - Click the 📎 icon to upload PDF files
    3. **Add Knowledge** - Use Manage Knowledge tab to add custom FAQs
    4. **Configure Settings** - Use sidebar to adjust AI behavior
    
    ### Setup:
    1. Replace `GEMINI_API_KEY` with your actual Gemini API key
    2. Make sure `faq.xlsx` file exists in the same directory
    3. Install required packages: `pip install streamlit sentence-transformers pandas google-generativeai PyPDF2 pdfplumber pillow pytesseract openpyxl`
    
    ### Supported Document Types:
    - Receipts and invoices
    - Reports and documents
    - Any text-based PDF
    
    ### Chat Features:
    - Maintains conversation history
    - References uploaded documents
    - Searches FAQ database
    - Provides source information
    """)
import streamlit as st
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
import hashlib

# Hardcode your Gemini API key here
GEMINI_API_KEY = "AIzaSyBlJspbf9LcUBcHuercP8Lyn_-n7uQX0E"  # Replace with your actual Gemini API key

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
if 'user_exists' not in st.session_state:
    st.session_state.user_exists = False

# User management functions
def generate_user_id(identifier):
    """Generate a consistent user ID from identifier (email/phone)"""
    return hashlib.md5(identifier.encode()).hexdigest()[:12]

def check_user_exists(identifier):
    """Check if user already exists based on email/phone"""
    user_file = "users.json"
    
    if os.path.exists(user_file):
        try:
            with open(user_file, 'r') as f:
                users = json.load(f)
            user_id = generate_user_id(identifier)
            return user_id in users, user_id
        except:
            return False, generate_user_id(identifier)
    return False, generate_user_id(identifier)

def save_user(identifier, user_data):
    """Save user information"""
    user_file = "users.json"
    users = {}
    
    if os.path.exists(user_file):
        try:
            with open(user_file, 'r') as f:
                users = json.load(f)
        except:
            users = {}
    
    user_id = generate_user_id(identifier)
    users[user_id] = {
        'identifier': identifier,
        'created_at': datetime.now().isoformat(),
        'last_active': datetime.now().isoformat(),
        **user_data
    }
    
    with open(user_file, 'w') as f:
        json.dump(users, f, indent=2)
    
    return user_id

# LOAD ALL FAQ DATA - No limits, load everything
@st.cache_data(show_spinner=False)
def load_faq_data(faq_path):
    """Load ALL FAQ data as simple text format - no limits"""
    if os.path.exists(faq_path):
        df = pd.read_excel(faq_path, engine='openpyxl')
        df = df.dropna(subset=['Question', 'Answer'])
        
        # Convert ALL entries to simple text format for long context
        faq_text = ""
        for i, row in df.iterrows():
            faq_text += f"FAQ {i+1}:\nQ: {row['Question']}\nA: {row['Answer']}\n\n"
        
        return faq_text, df
    else:
        return "", pd.DataFrame(columns=['Question', 'Answer'])

# PDF text extraction functions
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using multiple methods for better accuracy"""
    text = ""
    
    try:
        # Method 1: Try pdfplumber first
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # If pdfplumber didn't extract much text, try PyPDF2
        if len(text.strip()) < 50:
            pdf_file.seek(0)
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
            r'^([A-Z\s&]+)(?=\n|\r)',
            r'(.*?)(?=\n.*\d{3}[-.]?\d{3}[-.]?\d{4})'
        ]
    }
    
    for field, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                receipt_info[field] = matches[0] if isinstance(matches[0], str) else matches[0][0] if matches[0] else ""
                break
    
    return receipt_info

# Generate AI response using ALL FAQ data in long context
def generate_ai_response(user_question, faq_text, system_prompt, selected_model, chat_history, pdf_content=""):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "my key":
        return "Gemini API key not configured. Please check your API key."
    
    # Prepare recent chat history (last 2 exchanges)
    history_context = ""
    is_first_message = len(chat_history) == 0
    if chat_history:
        recent_history = chat_history[-2:]
        history_context = "\n".join([f"User: {h['user']}\nPriya: {h['assistant']}" 
                                   for h in recent_history if h.get('type') != 'system'])
    
    # Prepare PDF context (if any)
    pdf_context = ""
    if pdf_content:
        pdf_context = f"\n\nUPLOADED DOCUMENT:\n{pdf_content[:1500]}..."
    
    # LONG CONTEXT PROMPT - ALL FAQs included directly
    full_prompt = f"""{system_prompt}

COMPLETE KNOWLEDGE BASE (All Available FAQs):
{faq_text}

RECENT CONVERSATION:
{history_context}

{pdf_context}

CURRENT QUESTION: {user_question}

Instructions:
- Answer based on the complete FAQ knowledge base above
- If no relevant FAQ exists, use your general knowledge about Narayan Seva Sansthan
- Keep response short and clear (2-3 sentences max)
- Be helpful and accurate
- {"Greet with 'Jai Siyaram! I'm Priya'" if is_first_message else "Continue conversation naturally"}

Response:"""
    
    try:
        # Initialize Gemini model - use Pro for better handling of large context
        model_name = "gemini-1.5-pro" if selected_model == "gemini-pro" else "gemini-1.5-flash"
        gemini_model = genai.GenerativeModel(model_name)
        
        # Generate response with controlled creativity
        response = gemini_model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=200,
                top_p=0.8,
                top_k=40
            )
        )
        
        return response.text
    except Exception as e:
        return f"Error generating AI response: {str(e)}"

# -------------- Streamlit UI ----------------
st.title("📚 Complete FAQ Chatbot - All FAQs Loaded")

# USER REGISTRATION/LOGIN CHECK
if not st.session_state.user_exists:
    st.subheader("Welcome! Please provide your details to continue")
    
    with st.form("user_registration"):
        identifier = st.text_input("Email or Phone Number:", placeholder="your.email@example.com or +1234567890")
        name = st.text_input("Your Name:", placeholder="Full Name")
        submitted = st.form_submit_button("Continue")
        
        if submitted and identifier and name:
            user_exists, user_id = check_user_exists(identifier)
            
            if user_exists:
                st.success(f"Welcome back! User ID: {user_id}")
            else:
                save_user(identifier, {"name": name})
                st.success(f"New user registered! User ID: {user_id}")
            
            st.session_state.user_exists = True
            st.session_state.user_id = user_id
            st.session_state.user_name = name
            st.rerun()
    
    st.stop()

# Custom CSS for better styling
st.markdown("""
<style>
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
    
    # Show user info
    if st.session_state.user_exists:
        st.info(f"User: {st.session_state.get('user_name', 'Unknown')}\nID: {st.session_state.get('user_id', 'Unknown')}")
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="""You are Priya, an AI assistant at Narayan Seva Sansthan (NSS). 

Rules:
- Answer based on the provided complete FAQ knowledge base
- Be helpful, accurate, and concise
- Keep responses to 2-3 sentences maximum
- Don't make up information not in the knowledge base
- Use "Jai Siyaram!" greeting only for first interaction""",
        height=150
    )
    
    # Model selection - Default to Pro for better large context handling
    model_choice = st.selectbox(
        "Select Gemini Model",
        ["gemini-pro", "gemini-flash"],
        index=0,
        help="Pro recommended for handling complete FAQ database"
    )
    
    # Clear chat history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.processed_files = set()
        st.success("Chat history cleared!")
    
    # Logout button
    if st.button("🚪 Logout"):
        st.session_state.user_exists = False
        st.session_state.chat_history = []
        st.rerun()

# Load ALL FAQ data (no limits!)
faq_text, faq_df = load_faq_data(FAQ_FILE_PATH)

# Display FAQ loading status
if not faq_df.empty:
    st.sidebar.success(f"📊 Loaded ALL {len(faq_df)} FAQs successfully!")
    
    # Show context size estimation
    context_size = len(faq_text)
    st.sidebar.info(f"Context size: ~{context_size:,} characters")
    
    if context_size > 100000:
        st.sidebar.warning("⚠️ Large context - consider using Gemini Pro for better performance")
else:
    st.sidebar.error("❌ No FAQ file found! Please add faq.xlsx")

# Function to handle file upload and processing
def process_uploaded_file(uploaded_file):
    """Process uploaded PDF file and update session state"""
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if file_id in st.session_state.processed_files:
            return False
        
        with st.spinner("📖 Processing PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
            
            if extracted_text:
                st.session_state.pdf_content = extracted_text
                st.session_state.pdf_filename = uploaded_file.name
                st.session_state.processed_files.add(file_id)
                
                receipt_info = parse_receipt_info(extracted_text)
                
                receipt_summary = f"📄 Uploaded document: {uploaded_file.name}"
                if receipt_info:
                    receipt_details = []
                    for key, value in receipt_info.items():
                        if value:
                            receipt_details.append(f"{key.title()}: {value}")
                    if receipt_details:
                        receipt_summary += f"\n🧾 Detected: {', '.join(receipt_details[:3])}"
                
                st.session_state.chat_history.append({
                    'user': f"[Uploaded: {uploaded_file.name}]",
                    'assistant': receipt_summary,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'system'
                })
                
                return True
            else:
                st.error("❌ Could not extract text from the PDF.")
                return False
    return False

# Main interface tabs
tab1, tab2 = st.tabs(["💬 Chat", "📋 Complete Knowledge Base"])

with tab1:
    st.subheader("Chat with Priya - Full FAQ Knowledge Available")
    
    # Display chat history
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history):
            if chat.get('type') == 'system':
                st.info(chat['assistant'])
            else:
                col1, col2 = st.columns([1, 10])
                with col2:
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {chat["user"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-message assistant-message"><strong>Priya:</strong> {chat["assistant"]}</div>', unsafe_allow_html=True)
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Chat input
    col1, col2 = st.columns([9, 1])
    
    with col1:
        input_key = f"chat_input_{len(st.session_state.chat_history)}"
        user_question = st.text_input("Ask a question:", key=input_key, label_visibility="collapsed", placeholder="Type your message here...")
    
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Handle file upload
    if uploaded_file is not None:
        if process_uploaded_file(uploaded_file):
            st.success(f"✅ Processed: {uploaded_file.name}")
            st.rerun()
    
    # Handle text input
    if user_question and send_button:
        with st.spinner("🤔 Priya is thinking..."):
            # Generate AI response using ALL FAQ data in long context
            ai_response = generate_ai_response(
                user_question, 
                faq_text,  # Pass ALL FAQ text directly
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
            
            st.rerun()
    
    # Current document status
    if st.session_state.pdf_filename:
        st.sidebar.success(f"📄 Document: {st.session_state.pdf_filename}")
        if st.sidebar.button("🗑️ Remove Document"):
            st.session_state.pdf_content = ""
            st.session_state.pdf_filename = ""
            st.rerun()

with tab2:
    st.subheader("📋 Complete Knowledge Base")
    
    if not faq_df.empty:
        st.success(f"📊 Successfully loaded ALL {len(faq_df)} FAQs using Long Context approach!")
        
        # Search through FAQs
        search_query = st.text_input("🔍 Search FAQs:", placeholder="Type to search questions and answers...")
        
        if search_query:
            # Filter FAQs based on search
            mask = faq_df['Question'].str.contains(search_query, case=False, na=False) | \
                   faq_df['Answer'].str.contains(search_query, case=False, na=False)
            filtered_df = faq_df[mask]
            st.info(f"Found {len(filtered_df)} FAQs matching '{search_query}'")
        else:
            filtered_df = faq_df
        
        # Show loaded FAQs with pagination
        items_per_page = 10
        total_pages = (len(filtered_df) - 1) // items_per_page + 1
        
        if total_pages > 1:
            page = st.selectbox("Select page:", range(1, total_pages + 1))
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            page_df = filtered_df.iloc[start_idx:end_idx]
        else:
            page_df = filtered_df
        
        st.markdown(f"### Showing FAQs {len(page_df)} of {len(filtered_df)} total:")
        
        for i, (idx, row) in enumerate(page_df.iterrows()):
            with st.expander(f"FAQ {idx+1}: {row['Question'][:80]}{'...' if len(row['Question']) > 80 else ''}"):
                st.write(f"**Q:** {row['Question']}")
                st.write(f"**A:** {row['Answer']}")
                
                # Show character count for context estimation
                qa_length = len(row['Question']) + len(row['Answer'])
                st.caption(f"Length: {qa_length} characters")
    else:
        st.error("❌ No FAQ file found. Please add faq.xlsx to your directory.")
        st.info("The file should have columns: 'Question' and 'Answer'")
    
    # Add new FAQ manually
    with st.form("add_faq"):
        st.write("**Add New FAQ Entry**")
        new_q = st.text_input("Question:")
        new_a = st.text_area("Answer:")
        submitted = st.form_submit_button("Add to Knowledge Base")
        
        if submitted and new_q and new_a:
            try:
                # Append to existing Excel file
                new_row = pd.DataFrame({'Question': [new_q], 'Answer': [new_a]})
                if os.path.exists(FAQ_FILE_PATH):
                    existing_df = pd.read_excel(FAQ_FILE_PATH)
                    updated_df = pd.concat([existing_df, new_row], ignore_index=True)
                else:
                    updated_df = new_row
                
                updated_df.to_excel(FAQ_FILE_PATH, index=False)
                st.success("✅ New FAQ added successfully! Refresh the page to see it.")
            except Exception as e:
                st.error(f"Error adding FAQ: {str(e)}")

# Instructions
with st.expander("📋 Complete FAQ System Overview"):
    st.markdown(f"""
    ### ✨ Complete FAQ Loading System:
    
    **📊 Current Status:**
    - 📁 FAQ File: `{FAQ_FILE_PATH}`
    - 📝 Total FAQs Loaded: **{len(faq_df) if not faq_df.empty else 0}**
    - 📏 Total Context Size: **~{len(faq_text):,} characters**
    - 🤖 Recommended Model: **Gemini Pro** (for large contexts)
    
    **🚀 Key Features:**
    - ✅ **ALL FAQs loaded** - No limits or truncation
    - ✅ **Smart context management** - Optimized for large datasets
    - ✅ **Search functionality** - Find specific FAQs quickly
    - ✅ **Pagination** - Easy browsing of large FAQ collections
    - ✅ **Real-time addition** - Add new FAQs directly
    
    **🎯 Benefits of Loading All FAQs:**
    - **Complete Coverage:** AI has access to entire knowledge base
    - **Better Accuracy:** No relevant information is missed
    - **Contextual Understanding:** AI can see relationships between FAQs
    - **No Retrieval Errors:** No risk of missing relevant but "dissimilar" entries
    
    **💡 Performance Tips:**
    - Use **Gemini Pro** for FAQ sets > 50 entries
    - Monitor context size in sidebar
    - Consider chunking if FAQ file becomes very large (>500 entries)
    
    **🔧 Technical Details:**
    - Uses Gemini's long context window (up to 1M tokens)
    - No embeddings or vector search needed
    - Direct text inclusion in prompt
    - Efficient caching with `@st.cache_data`
    """)
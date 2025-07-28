import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from groq import Groq
import os
import PyPDF2
import pdfplumber
from PIL import Image
import pytesseract
import io
import re
from datetime import datetime
import json

# Hardcode your Groq API key here
API_KEY = "replace"  # Replace with your actual API key

# Hardcode the FAQ file path
FAQ_FILE_PATH = "faq.xlsx"  # Make sure this file is in the same directory as your script

groq_client = Groq(api_key=API_KEY)

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

# Generate AI response with chat history and PDF context
def generate_ai_response(user_question, relevant_context, system_prompt, selected_model, chat_history, pdf_content=""):
    if not groq_client:
        return "Groq client not initialized. Please check your API key."
    
    # Prepare context from retrieved FAQs
    faq_context = ""
    if not relevant_context.empty:
        faq_context = "\n\n".join([f"Q: {row['Question']}\nA: {row['Answer']}" 
                              for _, row in relevant_context.iterrows()])
    
    # Prepare chat history context
    history_context = ""
    if chat_history:
        recent_history = chat_history[-5:]  # Last 5 exchanges
        history_context = "\n".join([f"User: {h['user']}\nPriya: {h['assistant']}" 
                                   for h in recent_history])
    
    # Prepare PDF context
    pdf_context = ""
    if pdf_content:
        pdf_context = f"\n\nUploaded Document Content:\n{pdf_content[:2000]}..."  # Limit to 2000 chars
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": f"""You are Priya, a helpful assistant at Narayan Seva Sansthan (NSS). 

Recent Chat History:
{history_context}

FAQ Context:
{faq_context}

{pdf_context}

Current User Question: {user_question}

Please provide a helpful response based on all available context. If you're referencing information from the uploaded document, mention it clearly. Maintain conversation continuity with the chat history."""
        }
    ]
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=selected_model,
            temperature=0.7,
            max_tokens=300
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating AI response: {str(e)}"

# -------------- Streamlit UI ----------------
st.title("📚 Enhanced FAQ System with PDF Processing & Chat History")

# Sidebar for configuration
with st.sidebar:
    st.header("🤖 Configuration")
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="""You are Priya, a helpful AI assistant working at Narayan Seva Sansthan (NSS). You answer questions based on provided FAQ context, uploaded documents, and chat history. 
You should:
- Introduce yourself as Priya from NSS when greeting new users
- Provide accurate answers based on the given context
- Reference uploaded documents when relevant
- Maintain conversation continuity
- Be concise but comprehensive
- Use a friendly and professional tone""",
        height=200
    )
    
    # Model selection
    model_choice = st.selectbox(
        "Select Groq Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        index=0
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
                    # Regular chat messages
                    with st.container():
                        st.markdown(f"**You:** {chat['user']}")
                        st.markdown(f"**Priya:** {chat['assistant']}")
                        if i < len(st.session_state.chat_history) - 1:
                            st.markdown("---")
    
    # Chat input area with upload button
    st.markdown("---")
    
    # Create columns for input and buttons
    col1, col2, col3 = st.columns([7, 0.8, 1])
    
    with col1:
        user_question = st.text_input("Ask a question or upload a PDF:", key="chat_input", label_visibility="collapsed", placeholder="Type your message here...")
    
    with col2:
        # Create a simple file uploader with minimal styling
        uploaded_file = st.file_uploader(
            "",  # Empty label
            type="pdf",
            help="Upload PDF",
            label_visibility="collapsed",
            key="file_upload"
        )
        
        # Apply custom CSS to make it look like a button
        st.markdown("""
        <style>
        /* Target the file uploader container */
        .stFileUploader {
            width: 100% !important;
        }
        
        /* Hide the drag and drop text and make it look like a button */
        .stFileUploader > div > div {
            border: 1px solid #d1d5db !important;
            border-radius: 6px !important;
            background: white !important;
            padding: 0 !important;
            height: 38px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* Hide the drag and drop text */
        .stFileUploader > div > div > div > div {
            display: none !important;
        }
        
        /* Style the browse files button to look like our upload button */
        .stFileUploader > div > div > button {
            border: none !important;
            background: transparent !important;
            color: #374151 !important;
            font-size: 16px !important;
            padding: 0 !important;
            height: 100% !important;
            width: 100% !important;
        }
        
        /* Add our paperclip icon */
        .stFileUploader > div > div > button:before {
            content: "📎";
            font-size: 16px;
        }
        
        /* Hide the original button text */
        .stFileUploader > div > div > button > div {
            display: none !important;
        }
        
        /* Hover effects */
        .stFileUploader > div > div:hover {
            background-color: #f9fafb !important;
            border-color: #9ca3af !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    with col3:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Handle file upload
    if uploaded_file is not None:
        process_uploaded_file(uploaded_file)
    
    # Handle text input
    if user_question and send_button and user_question != st.session_state.last_question:
        # Store the current question to avoid reprocessing
        st.session_state.last_question = user_question
        
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
            
            # Rerun to refresh the chat display
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
    
    ### How to Use:
    1. **Start Chatting** - Go to Chat tab and ask questions
    2. **Upload Documents** - Use PDF Upload tab for document-based queries
    3. **Add Knowledge** - Use Manage Knowledge tab to add custom FAQs
    4. **Configure Settings** - Use sidebar to adjust AI behavior
    
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
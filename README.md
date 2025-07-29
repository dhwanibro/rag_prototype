# Enhanced FAQ System with PDF Processing & Chat History

A smart FAQ chatbot powered by Google's Gemini AI, designed for Narayan Seva Sansthan (NSS). Features PDF document processing, semantic search, and contextual conversations with Priya, your AI assistant.

## 🚀 Features

- **Conversational AI**: Chat with Priya, an AI assistant from Narayan Seva Sansthan
- **PDF Processing**: Upload and analyze PDF documents (receipts, invoices, reports)
- **Semantic Search**: Find relevant information using sentence transformers
- **Chat History**: Maintains conversation context across messages
- **Knowledge Management**: Add custom FAQ entries dynamically
- **Receipt Parsing**: Automatically extract key information from receipts
- **Gemini AI Integration**: Powered by Google's advanced language models

## 📋 Prerequisites

- Python 3.7+
- Google Gemini API key
- Required Python packages (see requirements.txt)

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a Gemini API key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a free API key
   - Copy the key for the next step

4. **Configure API key:**
   - Open the main Python file
   - Replace `GEMINI_API_KEY = "your_gemini_api_key_here"` with your actual API key

5. **Prepare FAQ file:**
   - Create an Excel file named `faq.xlsx` in the same directory
   - Include columns: `Question` and `Answer`
   - Or the app will create an empty knowledge base

## 🚀 Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`

3. **Start chatting:**
   - Type questions in the chat input
   - Upload PDF documents for analysis
   - Add new FAQ entries through the management tab

## 💬 How to Use

### Chat Interface
- **Type questions** naturally in the input field
- **Upload PDFs** using the file uploader above the chat
- **View chat history** with contextual responses
- **Clear conversation** using the sidebar button

### PDF Processing
- Supports text-based PDF documents
- Automatically extracts and parses content
- Detects receipt information (dates, amounts, items)
- Content becomes available for AI responses

### Knowledge Management
- Add custom FAQ entries through the "Manage Knowledge" tab
- View all existing FAQ entries
- Dynamic knowledge base updates

## ⚙️ Configuration

### Sidebar Options
- **System Prompt**: Customize Priya's behavior and personality
- **Gemini Model**: Choose between Flash (faster) or Pro (more capable)
- **Top K Results**: Adjust number of relevant FAQ matches
- **Clear History**: Reset conversation and uploaded files

### Model Options
- **Gemini Flash**: Faster responses, good for general queries
- **Gemini Pro**: More capable, better for complex reasoning

## 📁 File Structure

```
project/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── faq.xlsx           # FAQ knowledge base (optional)
└── README.md          # This file
```

## 🔧 Technical Details

### Core Components
- **Streamlit**: Web interface framework
- **Sentence Transformers**: Semantic similarity search
- **Google Generative AI**: Gemini API integration
- **PyPDF2 & pdfplumber**: PDF text extraction
- **Pandas**: Data handling and FAQ management

### AI Features
- Contextual conversation memory
- Semantic search with similarity scoring
- Dynamic knowledge base updates
- Receipt parsing and information extraction
- Multilingual support through Gemini

## 🚨 Troubleshooting

### Common Issues

**API Key Error:**
- Ensure your Gemini API key is correctly set in the code
- Check API key permissions and quotas

**PDF Processing Issues:**
- Ensure PDFs contain extractable text (not scanned images)
- Try different PDF files if extraction fails
- Check file size limits (default: 200MB)

**Dependencies Error:**
- Run `pip install -r requirements.txt` again
- Check Python version compatibility
- Consider using a virtual environment

**Streamlit Issues:**
- Clear browser cache
- Restart the Streamlit server
- Check port availability (8501)

## 🔐 Security Notes

- API keys are stored in the code (not recommended for production)
- For production deployment, use environment variables
- Uploaded files are processed in memory only
- No data is permanently stored

## 📝 Customization

### Modify Priya's Personality
Edit the system prompt in the sidebar to change how Priya responds:
- Tone and style
- Specific knowledge areas
- Response format preferences

### Add Your Organization
- Replace "Narayan Seva Sansthan" references
- Update contact information
- Modify branding and colors

### Extend Functionality
- Add new document types
- Implement user authentication
- Add database storage
- Create custom parsing rules

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

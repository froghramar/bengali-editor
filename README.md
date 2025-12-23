# Bengali Text Editor with AI Auto-completion

A modern Bengali text editor powered by AI with intelligent auto-completion, voice input, and document analysis capabilities. Features smart transliteration, speech-to-text conversion, and image/PDF analysis using Gemini Vision.

## ✨ Features

- 🤖 AI-powered auto-completion using an AI language model
- 🔤 Smart transliteration (Banglish → Bengali) with autocompletion
- 🎤 Voice input with speech-to-text conversion (Google Cloud Speech-to-Text)
- 📄 Image/PDF analysis with Gemini Vision (automatic image optimization)
- 📊 Training data collection and export for ML models
- Context-aware suggestions with intelligent mode detection
- ⌨️ Keyboard navigation (↑↓ arrows, Enter/Tab to accept, Esc to close)
- 💾 Save/export documents
- 🎨 Modern dark-themed UI with two-column layout
- 📑 Tabbed output panel for analysis results (HTML Preview, Summary, Extracted Text)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git Bash (Windows)
- 2-3GB RAM
- Internet (first run only)

### Setup & Run

**1. Clone the repository:**
```bash
git clone <repository-url>
cd bengali-editor
```

**2. Setup Backend:**
```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**2a. Configure Environment Variables:**
```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your credentials (optional, only if using Gemini/Google Cloud services)
# Get your Gemini API key from: https://makersuite.google.com/app/apikey
# For Speech-to-Text and Vision features, you'll need Google Cloud credentials
```

The `.env` file allows you to configure which backend to use without setting environment variables manually.

**Note:** For voice input and vision analysis features, you need Google Cloud credentials:
- Set `GOOGLE_APPLICATION_CREDENTIALS` to point to your service account JSON file
- Or ensure Google Cloud SDK is configured with `gcloud auth application-default login`

**3. Setup Frontend:**
```bash
cd ../frontend
```

Create `frontend/index.html` from the frontend artifact.

**4. Run the application:**

Open two Git Bash terminals:

**Terminal 1 - Backend:**
```bash
cd backend
source .venv/Scripts/activate
uvicorn main:app --reload --port 8000
```
*Wait for: "Model loaded successfully!" (first run may take time to download)*

**Terminal 2 - Frontend:**
```bash
cd frontend
python -m http.server 3000
```

**5. Open browser:** http://localhost:3000

## 📁 Project Structure

```
bengali-editor/
├── backend/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration and environment variables
│   ├── schemas.py           # Pydantic models/schemas
│   ├── utils.py             # Utility functions
│   ├── requirements.txt
│   ├── .env                 # Environment variables (not in git)
│   ├── .env.example         # Example environment variables template
│   ├── services/            # AI/ML service implementations
│   │   ├── gemini.py        # Gemini service (completion, transliteration, vision)
│   │   ├── transformers.py  # Transformers model service
│   │   └── speech.py        # Google Cloud Speech-to-Text service
│   ├── models/              # Model management
│   │   └── loader.py        # Model loading and initialization
│   ├── routes/              # API route handlers
│   │   ├── complete.py      # Text completion endpoint
│   │   ├── transliterate.py # Transliteration endpoint
│   │   ├── speech.py        # Speech-to-text endpoint
│   │   └── vision.py        # Vision analysis endpoint
│   ├── .venv/              # Virtual environment
│   └── .gitignore
├── frontend/
│   ├── index.html          # Main HTML file
│   ├── config.js           # API configuration
│   ├── BengaliEditor.js    # Main React component
│   ├── components/         # React components
│   │   ├── Icons.js
│   │   ├── Header.js
│   │   ├── EditorArea.js
│   │   ├── SuggestionsDropdown.js
│   │   ├── FileUpload.js
│   │   ├── OutputPanel.js
│   │   ├── StatusBar.js
│   │   └── Instructions.js
│   ├── utils/              # Utility functions
│   │   ├── textUtils.js
│   │   ├── api.js
│   │   ├── visionApi.js
│   │   └── localStorage.js
│   └── hooks/              # Custom hooks
│       └── useVoiceRecording.js
├── .gitignore              # Git ignore rules
└── README.md
```

## 🎯 Usage

### Text Editing
1. Type Bengali text (min 2 characters)
2. Auto-completion appears automatically
3. Use **↑↓** to navigate suggestions
4. Press **Enter** or **Tab** to accept
5. Press **Esc** to close suggestions
6. Click **Save** to download document

### Voice Input
1. Click the **🎤 Voice** button to start recording
2. Speak in Bengali
3. Click again to stop recording
4. The transcribed text will be automatically appended to your document

### Image/PDF Analysis
1. Click **📁 Upload Image/PDF** button
2. Select an image (JPEG, PNG, GIF, WebP) or PDF file
3. Optionally add context/prompt in the editor
4. Click **🔍 Analyze** button
5. View results in the right panel:
   - **HTML Preview** tab: Structured HTML representation (default)
   - **Summary** tab: Text summary of extracted content
   - **Extracted Text** tab: Raw extracted text

## 🔧 API Endpoints

- `GET /` - Health check (shows active backend configuration)
- `POST /complete` - Text completion
- `POST /transliterate` - Banglish to Bengali transliteration
- `POST /speech-to-text` - Speech-to-text conversion (audio file → Bengali text)
- `POST /analyze-vision` - Image/PDF analysis using Gemini Vision

## 🤖 Backend Implementation Options

The backend supports two implementations that can be easily switched:

### Transformers Models (Default)
- Uses local Hugging Face models (BLOOM-560M for completion, mBART for transliteration)
- No API key required
- Runs entirely on your machine

### Gemini Flash Model
- Uses Google's Gemini 2.5 Flash via API
- Requires authentication: either `GEMINI_API_KEY` (API key) or `GOOGLE_APPLICATION_CREDENTIALS` (Vertex AI service account)
- Faster and potentially more accurate
- Requires internet connection
- Supports vision analysis for images and PDFs

### Switching Between Implementations

**Recommended: Use `.env` file**

The easiest way to configure the backend is using a `.env` file:

1. Copy `.env.example` to `.env`:
   ```bash
   cd backend
   cp .env.example .env
   ```

2. Edit `.env` and set your preferences:
   
   **For API Key authentication:**
   ```env
   GEMINI_API_KEY=your_api_key_here
   USE_GEMINI_COMPLETE=false
   USE_GEMINI_TRANSLITERATE=false
   ```
   
   **For Vertex AI Service Account (if you have a JSON key file):**
   ```env
   GOOGLE_APPLICATION_CREDENTIALS=backend/vertex-ai-key.json
   USE_GEMINI_COMPLETE=false
   USE_GEMINI_TRANSLITERATE=false
   ```
   
   Note: Use only ONE authentication method (either API key OR service account, not both).

3. The application will automatically load these settings when it starts.

**Alternative: Use environment variables directly**

You can also set environment variables manually:

**Use Gemini for completion only:**
```bash
export USE_GEMINI_COMPLETE=true
export GEMINI_API_KEY=your_api_key_here
uvicorn main:app --reload --port 8000
```

**Use Gemini for transliteration only:**
```bash
export USE_GEMINI_TRANSLITERATE=true
export GEMINI_API_KEY=your_api_key_here
uvicorn main:app --reload --port 8000
```

**Use Gemini for both:**
```bash
export USE_GEMINI_COMPLETE=true
export USE_GEMINI_TRANSLITERATE=true
export GEMINI_API_KEY=your_api_key_here
uvicorn main:app --reload --port 8000
```

**Use Transformers (default):**
```bash
# No environment variables needed, or explicitly set:
export USE_GEMINI_COMPLETE=false
export USE_GEMINI_TRANSLITERATE=false
uvicorn main:app --reload --port 8000
```

**Windows (Git Bash):**
```bash
export USE_GEMINI_COMPLETE=true
export GEMINI_API_KEY=your_api_key_here
uvicorn main:app --reload --port 8000
```

**Windows (PowerShell):**
```powershell
$env:USE_GEMINI_COMPLETE="true"
$env:GEMINI_API_KEY="your_api_key_here"
uvicorn main:app --reload --port 8000
```

**Check active configuration:**
```bash
curl http://localhost:8000/
```

**Note:** Environment variables set in your shell take precedence over `.env` file values. This allows you to override `.env` settings when needed.


**Test with curl (in Git Bash):**

Completion:
```bash
curl -X POST http://localhost:8000/complete \
  -H "Content-Type: application/json" \
  -d '{"text": "আমি ভাত", "max_suggestions": 5}'
```

Transliteration:
```bash
curl -X POST http://localhost:8000/transliterate \
  -H "Content-Type: application/json" \
  -d '{"text": "ami tomake bhalobashi", "max_suggestions": 3}'
```

Vision Analysis:
```bash
curl -X POST http://localhost:8000/analyze-vision \
  -F "file=@path/to/image.jpg" \
  -F "prompt=Extract all text from this document"
```

## 🐛 Troubleshooting

### Backend won't start
```bash
# Recreate virtual environment
rm -rf backend/.venv
cd backend
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### Virtual environment activation fails
```bash
# In Git Bash on Windows, use:
source .venv/Scripts/activate

# Not .venv/bin/activate (that's for Linux/Mac)
```

### Model download fails
```bash
# Clear cache
rm -rf ~/.cache/huggingface/
python main.py
```

### Port 8000 already in use
```bash
# Find and kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use different port
uvicorn main:app --port 8001
```

### Python command not found in Git Bash
```bash
# Use 'python' instead of 'python3'
python --version

# Or add alias to ~/.bashrc
echo "alias python3=python" >> ~/.bashrc
```

### Voice input not working
```bash
# Check browser permissions for microphone
# Ensure HTTPS or localhost (browsers require secure context for microphone access)
# Check backend logs for Google Cloud Speech-to-Text errors
# Verify GOOGLE_APPLICATION_CREDENTIALS is set correctly
```

### Vision analysis fails
```bash
# Ensure Pillow and pdf2image are installed
pip install Pillow pdf2image

# For PDF support, you may need poppler:
# Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases
# Add poppler/bin to PATH
# Linux: sudo apt-get install poppler-utils
# Mac: brew install poppler

# Check Gemini API quota/limits
# Verify image file size (automatically optimized, but very large files may still fail)
```

## 🚀 Production Deployment

### Backend
```bash
pip install gunicorn
gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend
Deploy `index.html` to:
- Netlify
- Vercel
- GitHub Pages
- AWS S3

Update `API_URL` in `index.html` to your backend domain.

## 📚 Documentation

- **Backend artifact**: Full FastAPI code with AI model
- **Frontend artifact**: Complete HTML/React editor
- **API docs**: http://localhost:8000/docs (auto-generated when running)

## 🎓 Development Notes

### Code Structure
The project uses a modular architecture:

**Backend:**
- `main.py` - FastAPI app setup and route registration
- `services/` - Business logic for AI/ML services
- `routes/` - API endpoint handlers
- `models/` - Model loading and management
- `config.py` - Configuration management
- `schemas.py` - Pydantic models

**Frontend:**
- `BengaliEditor.js` - Main React component
- `components/` - Reusable UI components
- `utils/` - Utility functions and API calls
- `hooks/` - Custom React hooks

### Adding new features
- Backend: Add service in `services/`, route in `routes/`, update `main.py`
- Frontend: Create component in `components/`, add utility in `utils/` if needed

### Model configuration

**Transformers models:**
Adjust in `main.py`:
```python
outputs = model.generate(
    max_length=20,      # Longer suggestions
    temperature=0.8,    # Creativity (0.1-1.0)
    num_beams=10,       # Quality vs speed
)
```

**Gemini Flash:**
Configure via environment variables or modify prompts in `complete_with_gemini()` and `transliterate_with_gemini()` functions.

### Caching (optional)
```bash
pip install redis
# Add caching logic to main.py
```

## 🤝 Contributing

Areas for improvement:
- User dictionary
- Caching layer  
- Mobile app
- VS Code extension
- Fine-tune model on domain-specific text
- Multi-page PDF support
- Image annotation features

## 📄 License

MIT License

---

**Questions?** Check API docs at http://localhost:8000/docs when running
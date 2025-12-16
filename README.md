# Bengali Text Editor with AI Auto-completion

A modern Bengali text editor powered by an AI language model for intelligent auto-completion and context-aware suggestions.

## ✨ Features

- 🤖 AI-powered auto-completion using an AI language model
- Context-aware suggestions
- ⌨️ Keyboard navigation (↑↓ arrows, Enter, Esc)
- 💾 Save/export documents
- 🎨 Modern dark-themed UI

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

Create `backend/main.py` and `backend/requirements.txt` from the artifacts.

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
│   ├── main.py              # FastAPI + AI model
│   ├── requirements.txt
│   ├── .venv/              # Virtual environment
│   └── .gitignore
├── frontend/
│   └── index.html          # React editor UI
└── README.md
```

## 🎯 Usage

1. Type Bengali text (min 2 characters)
2. Auto-completion appears automatically
3. Use **↑↓** to navigate suggestions
4. Press **Enter** to accept
5. Press **Esc** to close suggestions
6. Click **Save** to download document

## 🔧 API Endpoints

- `GET /` - Health check
- `POST /complete` - Text completion
- `POST /complete-word` - Single word completion  


**Test with curl (in Git Bash):**
```bash
curl -X POST http://localhost:8000/complete \
  -H "Content-Type: application/json" \
  -d '{"text": "আমি ভাত", "max_suggestions": 5}'
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

### Adding new features
- Backend: Modify `main.py` endpoints
- Frontend: Update `index.html` JavaScript

### Model configuration
Adjust in `main.py`:
```python
outputs = model.generate(
    max_length=20,      # Longer suggestions
    temperature=0.8,    # Creativity (0.1-1.0)
    num_beams=10,       # Quality vs speed
)
```

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

## 📄 License

MIT License

---

**Questions?** Check API docs at http://localhost:8000/docs when running
// Instructions component
const Instructions = ({ API_URL }) => {
  return (
    <div className="bg-gray-800 border-t border-gray-700 px-6 py-3 text-xs text-gray-500">
      <p>
        <strong>Smart Mode:</strong> Automatically detects when to transliterate (English → Bengali) or auto-complete (Bengali text)
      </p>
      <p className="mt-1">
        <strong>Features:</strong> Toggle 🔤 Word or ✨ Sentence | 📊 Export training data for ML models | 🗑️ Clear stored data
      </p>
      <p className="mt-1">
        <strong>Shortcuts:</strong> ↑↓ Navigate | Enter/Tab: Accept | Esc: Close
      </p>
      <p className="mt-1"><strong>Backend:</strong> {API_URL} | Accepted suggestions are saved locally for training</p>
      <p className="mt-1"><strong>Voice Input:</strong> Click the 🎤 Voice button to record speech. The audio will be converted to Bengali text and appended to your document.</p>
    </div>
  );
};

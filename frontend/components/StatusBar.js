// Status bar component
const StatusBar = ({ text, currentMode, showSuggestions, enableTransliterate, enableComplete }) => {
  return (
    <div className="bg-gray-800 border-t border-gray-700 px-6 py-2 text-sm text-gray-400 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <span>Characters: {text.length}</span>
        <span>Words: {text.trim().split(/\s+/).filter(Boolean).length}</span>
        {currentMode && showSuggestions && (
          <span className="text-green-400">
            Mode: {currentMode === 'transliterate' ? '🔤 Transliterate' : '✨ Complete'}
          </span>
        )}
      </div>
      <div className="flex items-center gap-4">
        <span className={enableTransliterate ? 'text-blue-400' : 'text-gray-600'}>
          Word: {enableTransliterate ? 'ON' : 'OFF'}
        </span>
        <span className={enableComplete ? 'text-green-400' : 'text-gray-600'}>
          Sentence: {enableComplete ? 'ON' : 'OFF'}
        </span>
        <span>UTF-8</span>
        <span>Bengali</span>
      </div>
    </div>
  );
};

// Suggestions dropdown component
const SuggestionsDropdown = ({ 
  showSuggestions, 
  suggestions, 
  selectedIndex, 
  currentMode, 
  cursorPosition, 
  applySuggestion 
}) => {
  if (!showSuggestions || suggestions.length === 0) return null;

  return (
    <div
      className="absolute bg-gray-800 border border-gray-600 rounded-lg shadow-2xl overflow-hidden z-10"
      style={{
        top: `${cursorPosition.top + 30}px`,
        left: `${cursorPosition.left + 20}px`,
        minWidth: '200px',
        maxWidth: '400px'
      }}
    >
      <div className="px-3 py-1 bg-gray-700 text-xs text-gray-400 border-b border-gray-600">
        {currentMode === 'transliterate' ? '🔤 Transliteration' : '✨ Auto-complete'}
      </div>
      {suggestions.map((suggestion, index) => (
        <div
          key={index}
          onClick={() => applySuggestion(suggestion)}
          className={`px-4 py-2 cursor-pointer transition-colors ${
            index === selectedIndex
              ? 'bg-green-600 text-white'
              : 'hover:bg-gray-700'
          }`}
        >
          {suggestion}
        </div>
      ))}
    </div>
  );
};

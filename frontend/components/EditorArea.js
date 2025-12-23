// Editor area component
const EditorArea = ({
  text,
  handleTextChange,
  handleKeyDown,
  textareaRef,
  showSuggestions,
  suggestions,
  selectedIndex,
  currentMode,
  cursorPosition,
  applySuggestion,
  loading,
  isRecording
}) => {
  return (
    <div className="flex-1 relative overflow-hidden">
      <div className="absolute inset-0 p-6">
        <div className="relative h-full">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={handleTextChange}
            onKeyDown={handleKeyDown}
            placeholder="বাংলায় লিখুন... Type in Banglish (ami, tumi) or Bengali..."
            className="w-full h-full bg-gray-800 text-gray-100 p-4 rounded-lg border border-gray-700 focus:border-green-500 focus:outline-none resize-none text-lg"
            style={{ fontFamily: "'Noto Sans Bengali', sans-serif" }}
          />

          <SuggestionsDropdown
            showSuggestions={showSuggestions}
            suggestions={suggestions}
            selectedIndex={selectedIndex}
            currentMode={currentMode}
            cursorPosition={cursorPosition}
            applySuggestion={applySuggestion}
          />

          {/* Loading Indicator */}
          {loading && (
            <div className="absolute top-4 right-4 bg-gray-700 px-3 py-2 rounded-lg flex items-center gap-2">
              <LoaderIcon />
              <span className="text-sm">Loading...</span>
            </div>
          )}

          {/* Recording Indicator */}
          {isRecording && (
            <div className="absolute top-4 right-4 bg-red-600 px-3 py-2 rounded-lg flex items-center gap-2 animate-pulse">
              <div className="w-3 h-3 bg-white rounded-full"></div>
              <span className="text-sm text-white">Recording...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

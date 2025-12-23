// Main BengaliEditor component
const BengaliEditor = () => {
  const { useState, useRef } = React;

  const [text, setText] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [cursorPosition, setCursorPosition] = useState({ top: 0, left: 0 });
  const [currentMode, setCurrentMode] = useState('');
  const [enableTransliterate, setEnableTransliterate] = useState(true);
  const [enableComplete, setEnableComplete] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessingVoice, setIsProcessingVoice] = useState(false);
  
  const textareaRef = useRef(null);
  const debounceTimer = useRef(null);
  const abortController = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Voice recording functions
  const { toggleRecording } = window.useVoiceRecording(
    setIsRecording,
    setIsProcessingVoice,
    audioChunksRef,
    mediaRecorderRef,
    text,
    setText,
    textareaRef
  );

  const handleTextChange = (e) => {
    const newText = e.target.value;
    setText(newText);

    // Cancel any ongoing API request
    if (abortController.current) {
      abortController.current.abort();
    }

    // Clear previous debounce timer
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }

    // Get cursor position for suggestion dropdown
    const textarea = textareaRef.current;
    if (textarea) {
      const cursorPos = textarea.selectionStart;
      const textBeforeCursor = newText.substring(0, cursorPos);
      const lines = textBeforeCursor.split('\n');
      const currentLine = lines.length;
      const currentCol = lines[lines.length - 1].length;
      
      setCursorPosition({
        top: currentLine * 24,
        left: Math.min(currentCol * 10, 400)
      });
    }

    // Debounce API calls (wait 300ms after user stops typing)
    if (newText.trim().length > 0) {
      debounceTimer.current = setTimeout(() => {
        if (window.apiUtils && window.apiUtils.getSuggestions) {
          window.apiUtils.getSuggestions(
            newText,
            textareaRef,
            enableTransliterate,
            enableComplete,
            abortController,
            setLoading,
            setSuggestions,
            setShowSuggestions,
            setSelectedIndex,
            setCurrentMode
          );
        }
      }, 300);
    } else {
      setShowSuggestions(false);
      setSuggestions([]);
    }
  };

  const handleKeyDown = (e) => {
    if (!showSuggestions || suggestions.length === 0) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1) % suggestions.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev - 1 + suggestions.length) % suggestions.length);
    } else if (e.key === 'Enter' && showSuggestions) {
      e.preventDefault();
      applySuggestion(suggestions[selectedIndex]);
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
    } else if (e.key === 'Tab' && showSuggestions) {
      e.preventDefault();
      applySuggestion(suggestions[selectedIndex]);
    }
  };

  const applySuggestion = (suggestion) => {
    const textarea = textareaRef.current;
    const cursorPos = textarea.selectionStart;
    const textBefore = text.substring(0, cursorPos);
    const textAfter = text.substring(cursorPos);
    
    let newText;
    let newCursorPos;
    let originalInput;
    
    if (currentMode === 'transliterate') {
      // Replace only the current word being typed
      const words = textBefore.split(/[\s\n]+/);
      const currentWord = words[words.length - 1] || '';
      originalInput = currentWord;
      
      // Find where the current word starts
      const currentWordStart = textBefore.lastIndexOf(currentWord);
      
      // Replace current word with suggestion
      const beforeWord = text.substring(0, currentWordStart);
      newText = beforeWord + suggestion + textAfter;
      newCursorPos = currentWordStart + suggestion.length;
      
      // Save transliteration data for training
      if (window.localStorageUtils && window.localStorageUtils.saveTransliterationData) {
        window.localStorageUtils.saveTransliterationData(originalInput, suggestion);
      }
    } else {
      // Auto-completion mode: add suggestion after current text
      originalInput = textBefore.trim();
      newText = textBefore + ' ' + suggestion + textAfter;
      newCursorPos = textBefore.length + suggestion.length + 1;
      
      // Save completion data for training
      if (window.localStorageUtils && window.localStorageUtils.saveCompletionData) {
        window.localStorageUtils.saveCompletionData(originalInput, suggestion);
      }
    }
    
    setText(newText);
    setShowSuggestions(false);
    
    // Set cursor position after inserted text
    setTimeout(() => {
      textarea.setSelectionRange(newCursorPos, newCursorPos);
      textarea.focus();
    }, 0);
  };

  const saveFile = () => {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'bengali-document.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-gray-100">
      <Header
        enableTransliterate={enableTransliterate}
        setEnableTransliterate={setEnableTransliterate}
        enableComplete={enableComplete}
        setEnableComplete={setEnableComplete}
        exportTrainingData={window.localStorageUtils?.exportTrainingData}
        clearTrainingData={window.localStorageUtils?.clearTrainingData}
        toggleRecording={toggleRecording}
        isRecording={isRecording}
        isProcessingVoice={isProcessingVoice}
        saveFile={saveFile}
      />

      <EditorArea
        text={text}
        handleTextChange={handleTextChange}
        handleKeyDown={handleKeyDown}
        textareaRef={textareaRef}
        showSuggestions={showSuggestions}
        suggestions={suggestions}
        selectedIndex={selectedIndex}
        currentMode={currentMode}
        cursorPosition={cursorPosition}
        applySuggestion={applySuggestion}
        loading={loading}
        isRecording={isRecording}
      />

      <StatusBar
        text={text}
        currentMode={currentMode}
        showSuggestions={showSuggestions}
        enableTransliterate={enableTransliterate}
        enableComplete={enableComplete}
      />

      <Instructions API_URL={window.API_URL} />
    </div>
  );
};

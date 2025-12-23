// Text processing utilities

// Helper function to check if text is primarily English/Latin characters
const isEnglishText = (str) => {
  if (!str) return false;
  // Count Latin alphabet characters vs Bengali characters
  const latinChars = str.match(/[a-zA-Z]/g) || [];
  const bengaliChars = str.match(/[\u0980-\u09FF]/g) || [];
  
  // If more than 70% are Latin characters, consider it English
  return latinChars.length > bengaliChars.length && latinChars.length > str.length * 0.5;
};

// Intelligent mode detection
const detectMode = (inputText, textareaRef, enableTransliterate, enableComplete) => {
  const textarea = textareaRef.current;
  const cursorPos = textarea.selectionStart;
  const textBeforeCursor = inputText.substring(0, cursorPos);
  
  // Extract the current word (text after last space/newline)
  const words = textBeforeCursor.split(/[\s\n]+/);
  const currentWord = words[words.length - 1] || '';
  
  // Get the text before the current word
  const textBeforeCurrentWord = textBeforeCursor.substring(0, textBeforeCursor.lastIndexOf(currentWord));
  
  // If only one feature is enabled, always use that feature
  if (enableTransliterate && !enableComplete) {
    // Only transliterate is enabled - use it for everything
    if (currentWord.length >= 1) {
      return { mode: 'transliterate', text: currentWord };
    }
    return null;
  }
  
  if (!enableTransliterate && enableComplete) {
    // Only complete is enabled - use it for everything
    if (textBeforeCursor.trim().length > 2) {
      return { mode: 'complete', text: inputText };
    }
    return null;
  }
  
  // Both features enabled - use smart detection
  if (enableTransliterate && enableComplete) {
    // Decision logic:
    // 1. If current word is English/Latin and has at least 1 character -> transliterate
    // 2. If current word is Bengali or empty, and there's Bengali text before -> complete
    // 3. Otherwise, prefer transliterate for single English words
    
    if (currentWord.length >= 1 && isEnglishText(currentWord)) {
      return { mode: 'transliterate', text: currentWord };
    }
    
    // Check if there's Bengali content to complete
    const hasBengaliContent = textBeforeCurrentWord.match(/[\u0980-\u09FF]/g);
    if (hasBengaliContent && textBeforeCursor.trim().length > 2) {
      return { mode: 'complete', text: inputText };
    }
    
    // Default to transliterate if we have an English word
    if (currentWord.length >= 1) {
      return { mode: 'transliterate', text: currentWord };
    }
  }
  
  // Both features disabled
  return null;
};

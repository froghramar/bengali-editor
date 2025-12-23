// API utility functions

// Get completions or transliterations from backend
const getSuggestions = async (inputText, textareaRef, enableTransliterate, enableComplete, abortController, setLoading, setSuggestions, setShowSuggestions, setSelectedIndex, setCurrentMode) => {
  const { detectMode } = window.textUtils || {};
  const detection = detectMode(inputText, textareaRef, enableTransliterate, enableComplete);
  
  if (!detection) {
    setShowSuggestions(false);
    setSuggestions([]);
    return;
  }
  
  // Cancel previous request if it exists
  if (abortController.current) {
    abortController.current.abort();
  }
  
  // Create new abort controller for this request
  abortController.current = new AbortController();
  
  setLoading(true);
  try {
    const endpoint = detection.mode === 'transliterate' ? '/transliterate' : '/complete';
    
    const response = await fetch(`${window.API_URL}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        text: detection.text,
        max_suggestions: 5,
        max_length: 20
      }),
      signal: abortController.current.signal
    });
    
    if (!response.ok) {
      throw new Error('API request failed');
    }
    
    const data = await response.json();
    setSuggestions(data.suggestions || []);
    setShowSuggestions((data.suggestions || []).length > 0);
    setSelectedIndex(0);
    setCurrentMode(detection.mode);
  } catch (error) {
    // Ignore abort errors (these are expected when cancelling)
    if (error.name === 'AbortError') {
      console.log('Request cancelled');
      return;
    }
    console.error('Error fetching suggestions:', error);
    setSuggestions([]);
    setShowSuggestions(false);
  } finally {
    setLoading(false);
  }
};

// Send audio to backend for transcription
const sendAudioToBackend = async (audioBlob, API_URL, text, setText, textareaRef, setIsProcessingVoice) => {
  setIsProcessingVoice(true);
  try {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    
    const response = await fetch(`${API_URL}/speech-to-text`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || 'Failed to transcribe audio');
    }
    
    const data = await response.json();
    
    if (data.text) {
      // Append transcribed text to the textarea at cursor position
      const textarea = textareaRef.current;
      const cursorPos = textarea.selectionStart;
      const textBefore = text.substring(0, cursorPos);
      const textAfter = text.substring(cursorPos);
      
      // Add space before if there's text before cursor
      const spaceBefore = textBefore.trim().length > 0 ? ' ' : '';
      // Add space after if there's text after cursor
      const spaceAfter = textAfter.trim().length > 0 ? ' ' : '';
      
      const newText = textBefore + spaceBefore + data.text + spaceAfter + textAfter;
      setText(newText);
      
      // Set cursor position after inserted text
      const newCursorPos = cursorPos + spaceBefore.length + data.text.length + spaceAfter.length;
      setTimeout(() => {
        textarea.setSelectionRange(newCursorPos, newCursorPos);
        textarea.focus();
      }, 0);
    } else {
      alert('No text was transcribed from the audio.');
    }
  } catch (error) {
    console.error('Error sending audio to backend:', error);
    alert(`Error transcribing audio: ${error.message}`);
  } finally {
    setIsProcessingVoice(false);
  }
};

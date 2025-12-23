// Voice recording hook/utility

const useVoiceRecording = (setIsRecording, setIsProcessingVoice, audioChunksRef, mediaRecorderRef, text, setText, textareaRef) => {
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
        
        // Create audio blob
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        
        // Send to backend for transcription
        if (window.apiUtils && window.apiUtils.sendAudioToBackend) {
          await window.apiUtils.sendAudioToBackend(audioBlob, window.API_URL, text, setText, textareaRef, setIsProcessingVoice);
        }
      };
      
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Could not access microphone. Please check permissions.');
    }
  };

  const stopRecording = (isRecording) => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const toggleRecording = (isRecording) => {
    if (isRecording) {
      stopRecording(isRecording);
    } else {
      startRecording();
    }
  };

  return { startRecording, stopRecording, toggleRecording };
};

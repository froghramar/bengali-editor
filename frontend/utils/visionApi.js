// Vision analysis API utility

const analyzeVision = async (file, prompt, API_URL, setIsAnalyzing, setAnalysisResult) => {
  setIsAnalyzing(true);
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('prompt', prompt || '');
    
    const response = await fetch(`${API_URL}/analyze-vision`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || 'Failed to analyze file');
    }
    
    const data = await response.json();
    setAnalysisResult(data);
  } catch (error) {
    console.error('Error analyzing file:', error);
    alert(`Error analyzing file: ${error.message}`);
    setAnalysisResult(null);
  } finally {
    setIsAnalyzing(false);
  }
};

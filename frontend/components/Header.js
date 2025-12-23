// Header component
const Header = ({ 
  enableTransliterate, 
  setEnableTransliterate, 
  enableComplete, 
  setEnableComplete,
  exportTrainingData,
  clearTrainingData,
  toggleRecording,
  isRecording,
  isProcessingVoice,
  saveFile
}) => {
  return (
    <div className="bg-gray-800 border-b border-gray-700 px-6 py-4 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="text-green-400">
          <FileTextIcon />
        </div>
        <h1 className="text-xl font-semibold">বাংলা টেক্সট এডিটর</h1>
        <span className="ml-4 px-3 py-1 bg-purple-600 text-white text-xs rounded-full">
          Smart Mode
        </span>
      </div>
      <div className="flex items-center gap-3">
        {/* Feature Toggles */}
        <div className="flex gap-2 mr-4 border-r border-gray-600 pr-4">
          <button
            onClick={() => setEnableTransliterate(!enableTransliterate)}
            className={`px-3 py-1.5 rounded text-sm transition-colors flex items-center gap-2 ${
              enableTransliterate 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
            }`}
            title="Toggle word-level transliteration (Banglish → Bengali)"
          >
            <span className="text-xs">🔤</span>
            <span>Word</span>
          </button>
          <button
            onClick={() => setEnableComplete(!enableComplete)}
            className={`px-3 py-1.5 rounded text-sm transition-colors flex items-center gap-2 ${
              enableComplete 
                ? 'bg-green-600 text-white' 
                : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
            }`}
            title="Toggle sentence-level auto-completion"
          >
            <span className="text-xs">✨</span>
            <span>Sentence</span>
          </button>
        </div>
        
        {/* Training Data Controls */}
        <div className="flex gap-2 mr-4 border-r border-gray-600 pr-4">
          <button
            onClick={exportTrainingData}
            className="px-3 py-1.5 bg-purple-600 hover:bg-purple-700 rounded text-sm transition-colors flex items-center gap-2"
            title="Export training data for model training"
          >
            <span className="text-xs">📊</span>
            <span>Export Data</span>
          </button>
          <button
            onClick={clearTrainingData}
            className="px-3 py-1.5 bg-red-600 hover:bg-red-700 rounded text-sm transition-colors flex items-center gap-2"
            title="Clear all training data"
          >
            <span className="text-xs">🗑️</span>
            <span>Clear</span>
          </button>
        </div>
        
        <button
          onClick={() => toggleRecording(isRecording)}
          disabled={isProcessingVoice}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
            isRecording
              ? 'bg-red-600 hover:bg-red-700 animate-pulse'
              : isProcessingVoice
              ? 'bg-gray-600 cursor-not-allowed'
              : 'bg-orange-600 hover:bg-orange-700'
          }`}
          title={isRecording ? 'Click to stop recording' : 'Click to start voice input'}
        >
          <MicrophoneIcon />
          {isRecording ? 'Recording...' : isProcessingVoice ? 'Processing...' : 'Voice'}
        </button>
        
        <button
          onClick={saveFile}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
        >
          <SaveIcon />
          Save
        </button>
      </div>
    </div>
  );
};

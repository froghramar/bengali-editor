// File upload component
const FileUpload = ({ onFileSelect, selectedFile, onAnalyze, isAnalyzing }) => {
  const fileInputRef = React.useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file type
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp', 'application/pdf'];
      if (!validTypes.includes(file.type)) {
        alert('Please upload an image (JPEG, PNG, GIF, WebP) or PDF file.');
        return;
      }
      
      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB.');
        return;
      }
      
      onFileSelect(file);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
      <div className="flex items-center gap-4">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,application/pdf"
          onChange={handleFileChange}
          className="hidden"
        />
        <button
          onClick={handleUploadClick}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors text-sm"
        >
          📁 Upload Image/PDF
        </button>
        
        {selectedFile && (
          <div className="flex-1 flex items-center gap-2">
            <span className="text-sm text-gray-300">
              {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
            </span>
            <button
              onClick={() => onFileSelect(null)}
              className="text-red-400 hover:text-red-300 text-sm"
            >
              ✕
            </button>
          </div>
        )}
        
        {selectedFile && (
          <button
            onClick={onAnalyze}
            disabled={isAnalyzing}
            className={`px-4 py-2 rounded-lg transition-colors text-sm ${
              isAnalyzing
                ? 'bg-gray-600 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700'
            }`}
          >
            {isAnalyzing ? '⏳ Analyzing...' : '🔍 Analyze'}
          </button>
        )}
      </div>
    </div>
  );
};

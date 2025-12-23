// Output panel component for displaying analysis results
const OutputPanel = ({ analysisResult, isLoading }) => {
  if (isLoading) {
    return (
      <div className="h-full bg-gray-800 border-l border-gray-700 flex items-center justify-center">
        <div className="text-center">
          <LoaderIcon />
          <p className="mt-2 text-gray-400">Analyzing document...</p>
        </div>
      </div>
    );
  }

  if (!analysisResult) {
    return (
      <div className="h-full bg-gray-800 border-l border-gray-700 flex items-center justify-center">
        <div className="text-center text-gray-500">
          <p className="text-lg mb-2">📄 Document Analysis</p>
          <p className="text-sm">Upload an image or PDF and click Analyze to see results here</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full bg-gray-800 border-l border-gray-700 flex flex-col">
      {/* Summary Section */}
      <div className="border-b border-gray-700 p-4">
        <h3 className="text-lg font-semibold text-gray-200 mb-2">📋 Summary</h3>
        <div className="text-sm text-gray-300 whitespace-pre-wrap bg-gray-900 p-3 rounded max-h-48 overflow-y-auto">
          {analysisResult.summary}
        </div>
      </div>

      {/* HTML Preview Section */}
      <div className="flex-1 overflow-auto p-4">
        <h3 className="text-lg font-semibold text-gray-200 mb-2">👁️ HTML Preview</h3>
        <div className="bg-white rounded-lg p-4 overflow-auto max-h-full">
          <div 
            dangerouslySetInnerHTML={{ __html: analysisResult.html_output }}
            className="prose max-w-none"
          />
        </div>
      </div>

      {/* Extracted Text Section (if available) */}
      {analysisResult.extracted_text && (
        <div className="border-t border-gray-700 p-4 max-h-32 overflow-y-auto">
          <h3 className="text-sm font-semibold text-gray-200 mb-2">📝 Extracted Text</h3>
          <div className="text-xs text-gray-400 whitespace-pre-wrap bg-gray-900 p-2 rounded">
            {analysisResult.extracted_text}
          </div>
        </div>
      )}
    </div>
  );
};

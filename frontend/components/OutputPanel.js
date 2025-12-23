// Output panel component for displaying analysis results
const OutputPanel = ({ analysisResult, isLoading }) => {
  const [activeTab, setActiveTab] = React.useState('html');

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
      {/* Tabs */}
      <div className="flex border-b border-gray-700 bg-gray-900">
        <button
          onClick={() => setActiveTab('html')}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'html'
              ? 'bg-gray-800 text-white border-b-2 border-blue-500'
              : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800'
          }`}
        >
          👁️ HTML Preview
        </button>
        <button
          onClick={() => setActiveTab('summary')}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'summary'
              ? 'bg-gray-800 text-white border-b-2 border-blue-500'
              : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800'
          }`}
        >
          📋 Summary
        </button>
        {analysisResult.extracted_text && (
          <button
            onClick={() => setActiveTab('text')}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === 'text'
                ? 'bg-gray-800 text-white border-b-2 border-blue-500'
                : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800'
            }`}
          >
            📝 Extracted Text
          </button>
        )}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-auto">
        {activeTab === 'html' && (
          <div className="h-full p-4">
            <div className="bg-white rounded-lg p-4 overflow-auto h-full">
              <div 
                dangerouslySetInnerHTML={{ __html: analysisResult.html_output }}
                className="prose max-w-none"
              />
            </div>
          </div>
        )}

        {activeTab === 'summary' && (
          <div className="h-full p-4">
            <div className="text-sm text-gray-300 whitespace-pre-wrap bg-gray-900 p-4 rounded h-full overflow-y-auto">
              {analysisResult.summary}
            </div>
          </div>
        )}

        {activeTab === 'text' && analysisResult.extracted_text && (
          <div className="h-full p-4">
            <div className="text-sm text-gray-300 whitespace-pre-wrap bg-gray-900 p-4 rounded h-full overflow-y-auto">
              {analysisResult.extracted_text}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

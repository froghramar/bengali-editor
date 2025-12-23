// LocalStorage utilities for training data management

// Maintain an index of all storage keys for easy retrieval
const updateKeyIndex = (type, storageKey) => {
  try {
    const indexKey = `bengali_${type}_keys_index`;
    const existing = JSON.parse(localStorage.getItem(indexKey) || '[]');
    
    if (!existing.includes(storageKey)) {
      existing.push(storageKey);
      localStorage.setItem(indexKey, JSON.stringify(existing));
    }
  } catch (error) {
    console.error('Error updating key index:', error);
  }
};

// Get all keys for a specific type
const getAllKeys = (type) => {
  try {
    const indexKey = `bengali_${type}_keys_index`;
    return JSON.parse(localStorage.getItem(indexKey) || '[]');
  } catch (error) {
    console.error('Error getting keys:', error);
    return [];
  }
};

// Save transliteration training data
const saveTransliterationData = (input, output) => {
  try {
    // Generate hourly key: bengali_transliteration_2024-12-17T10
    const now = new Date();
    const hourKey = now.toISOString().slice(0, 13); // YYYY-MM-DDTHH
    const storageKey = `bengali_transliteration_${hourKey}`;
    
    const existing = JSON.parse(localStorage.getItem(storageKey) || '[]');
    
    const dataPoint = {
      input: input,
      output: output,
      timestamp: now.toISOString(),
      mode: 'transliterate'
    };
    
    existing.push(dataPoint);
    localStorage.setItem(storageKey, JSON.stringify(existing));
    
    // Also maintain an index of all keys for easy export
    updateKeyIndex('transliteration', storageKey);
    
    console.log('Saved transliteration data to:', storageKey);
  } catch (error) {
    console.error('Error saving transliteration data:', error);
  }
};

// Save completion training data
const saveCompletionData = (input, output) => {
  try {
    // Generate hourly key: bengali_completion_2024-12-17T10
    const now = new Date();
    const hourKey = now.toISOString().slice(0, 13); // YYYY-MM-DDTHH
    const storageKey = `bengali_completion_${hourKey}`;
    
    const existing = JSON.parse(localStorage.getItem(storageKey) || '[]');
    
    const dataPoint = {
      input: input,
      output: output,
      timestamp: now.toISOString(),
      mode: 'complete'
    };
    
    existing.push(dataPoint);
    localStorage.setItem(storageKey, JSON.stringify(existing));
    
    // Also maintain an index of all keys for easy export
    updateKeyIndex('completion', storageKey);
    
    console.log('Saved completion data to:', storageKey);
  } catch (error) {
    console.error('Error saving completion data:', error);
  }
};

// Export training data
const exportTrainingData = () => {
  try {
    // Get all transliteration data
    const transliterationKeys = getAllKeys('transliteration');
    const transliterationData = [];
    transliterationKeys.forEach(key => {
      const data = JSON.parse(localStorage.getItem(key) || '[]');
      transliterationData.push(...data);
    });
    
    // Get all completion data
    const completionKeys = getAllKeys('completion');
    const completionData = [];
    completionKeys.forEach(key => {
      const data = JSON.parse(localStorage.getItem(key) || '[]');
      completionData.push(...data);
    });
    
    const exportData = {
      transliteration: transliterationData,
      completion: completionData,
      exported_at: new Date().toISOString(),
      total_samples: {
        transliteration: transliterationData.length,
        completion: completionData.length
      },
      storage_keys: {
        transliteration: transliterationKeys,
        completion: completionKeys
      }
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bengali-training-data-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    alert(`Exported ${exportData.total_samples.transliteration} transliteration samples and ${exportData.total_samples.completion} completion samples from ${transliterationKeys.length + completionKeys.length} storage keys`);
  } catch (error) {
    console.error('Error exporting training data:', error);
    alert('Error exporting training data');
  }
};

// Clear training data
const clearTrainingData = () => {
  if (confirm('Are you sure you want to clear all training data? This cannot be undone.')) {
    try {
      // Clear all transliteration keys
      const transliterationKeys = getAllKeys('transliteration');
      transliterationKeys.forEach(key => localStorage.removeItem(key));
      localStorage.removeItem('bengali_transliteration_keys_index');
      
      // Clear all completion keys
      const completionKeys = getAllKeys('completion');
      completionKeys.forEach(key => localStorage.removeItem(key));
      localStorage.removeItem('bengali_completion_keys_index');
      
      alert(`Cleared ${transliterationKeys.length + completionKeys.length} storage keys`);
    } catch (error) {
      console.error('Error clearing training data:', error);
      alert('Error clearing training data');
    }
  }
};

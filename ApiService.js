const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const ApiService = {
  // Train the model
  trainModel: async () => {
    const response = await fetch(`${API_BASE_URL}/train`, {
      method: 'POST',
    });
    return response.json();
  },

  // Get recommendations
  getRecommendations: async (userData) => {
    const response = await fetch(`${API_BASE_URL}/recommend`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData),
    });
    return response.json();
  },

  // Get feature importance
  getFeatureImportance: async () => {
    const response = await fetch(`${API_BASE_URL}/feature-importance`);
    return response.json();
  },
}; 
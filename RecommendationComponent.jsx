import React, { useState, useEffect } from 'react';
import { ApiService } from './ApiService';

const RecommendationComponent = () => {
  const [recommendations, setRecommendations] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const getUserRecommendations = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Updated skill names to match exactly with the dataset
      const userData = {
        age: 25,
        education_level: 'Bachelor',
        career_goal: 'Data Science',
        preferred_learning_style: 'Visual',
        time_availability_hours_per_week: 10,
        learning_pace: 1.0,
        weekly_study_hours: 15,
        platform_visits_per_week: 5,
        engagement_score: 85,
        skill_python: 40,
        skill_statistics: 30,
        'skill_machine learning': 20, // Note the space
        skill_html_css: 0,
        skill_javascript: 0,
        skill_react: 0,
        'skill_social media': 0,  // Added missing skills
        skill_seo: 0,
        skill_analytics: 0
      };

      // First ensure model is trained
      await ApiService.trainModel();

      // Then get recommendations
      const response = await ApiService.getRecommendations(userData);
      if (response.status === 'success') {
        setRecommendations(response.data);
      } else {
        setError(response.message);
      }
    } catch (err) {
      setError('Failed to get recommendations: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <button 
        onClick={getUserRecommendations}
        disabled={isLoading}
      >
        {isLoading ? 'Loading...' : 'Get Recommendations'}
      </button>
      
      {error && <div className="error" style={{color: 'red'}}>{error}</div>}
      
      {recommendations && (
        <div>
          <h2>Recommendations</h2>
          <p>Predicted Success Rate: {recommendations.predicted_success_rate.toFixed(2)}%</p>
          
          <h3>Recommended Courses:</h3>
          <ul>
            {recommendations.recommended_courses.map((course, index) => (
              <li key={index}>{course}</li>
            ))}
          </ul>
          
          <h3>Skill Gaps:</h3>
          <ul>
            {recommendations.skill_gaps.map((gap, index) => (
              <li key={index}>
                {gap.skill}: Current Level - {gap.current_level.toFixed(2)}% (Difficulty: {gap.difficulty})
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default RecommendationComponent; 
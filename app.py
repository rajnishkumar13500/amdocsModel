from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import json
import ast

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

class CourseRecommender:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.is_trained = False

    def parse_course_history(self, history_str):
        if isinstance(history_str, str):
            try:
                return ast.literal_eval(history_str)
            except:
                return []
        return history_str if isinstance(history_str, list) else []

    def preprocess_data(self, df, training=True):
        data = df.copy()
        
        # Parse course history and extract features
        data['course_history'] = data['course_history'].apply(self.parse_course_history)
        
        # Calculate average completion rate and duration
        data['avg_course_completion'] = data['course_history'].apply(
            lambda x: np.mean([float(course.get('completion_rate', 0)) for course in x]) if x else 0
        )
        data['avg_course_duration'] = data['course_history'].apply(
            lambda x: np.mean([float(course.get('duration_days', 0)) for course in x]) if x else 0
        )

        # Handle categorical features
        categorical_features = ['education_level', 'career_goal', 'preferred_learning_style']
        for feature in categorical_features:
            if training:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    self.label_encoders[feature].fit(data[feature])
            data[feature] = self.label_encoders[feature].transform(data[feature])

        # Select features for training
        selected_features = [
            'age', 'education_level', 'career_goal', 'preferred_learning_style',
            'time_availability_hours_per_week', 'learning_pace', 'weekly_study_hours',
            'platform_visits_per_week', 'engagement_score', 'avg_course_completion',
            'avg_course_duration'
        ]

        # Update skill columns list to match exactly with dataset
        skill_columns = [
            'skill_python',
            'skill_statistics',
            'skill_machine learning',  # Note the space
            'skill_html_css',
            'skill_javascript',
            'skill_react',
            'skill_social media',
            'skill_seo',
            'skill_analytics'
        ]
        
        selected_features.extend(skill_columns)
        
        # Ensure all skill columns exist with default value 0
        for col in skill_columns:
            if col not in data.columns:
                data[col] = 0
        
        return data[selected_features]

    def train_model(self, df):
        try:
            # Prepare features and target
            X = self.preprocess_data(df)
            y = df['course_success_rate']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train Random Forest model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)

            # Calculate feature importance
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = importance
            self.is_trained = True

            return True

        except Exception as e:
            print(f"Error in training: {str(e)}")
            return False

    def recommend_courses(self, user_data):
        if not self.is_trained:
            raise ValueError("Model needs to be trained first")
        
        # Create a DataFrame with user data
        user_df = pd.DataFrame([user_data])
        
        # Ensure all required skill columns exist
        required_skills = [
            'skill_python',
            'skill_statistics',
            'skill_machine learning',
            'skill_html_css',
            'skill_javascript',
            'skill_react',
            'skill_social media',
            'skill_seo',
            'skill_analytics'
        ]
        
        for skill in required_skills:
            if skill not in user_df.columns:
                user_df[skill] = 0
        
        # Process user data
        processed_data = self.preprocess_data(user_df, training=False)
        
        try:
            # Scale features
            processed_data_scaled = self.scaler.transform(processed_data)
            
            # Predict success rate
            predicted_success = self.model.predict(processed_data_scaled)[0]
            
            # Generate recommendations based on user profile
            career_goal = user_data['career_goal']
            skill_levels = {k: v for k, v in user_data.items() if k.startswith('skill_')}
            
            recommended_courses = []
            skill_gaps = []
            
            if career_goal == 'Data Science':
                if skill_levels.get('skill_python', 0) < 60:
                    recommended_courses.append('Python Programming Fundamentals')
                    skill_gaps.append({
                        'skill': 'python',
                        'current_level': skill_levels.get('skill_python', 0),
                        'difficulty': 'beginner'
                    })
                if skill_levels.get('skill_statistics', 0) < 60:
                    recommended_courses.append('Statistics for Data Science')
                    skill_gaps.append({
                        'skill': 'statistics',
                        'current_level': skill_levels.get('skill_statistics', 0),
                        'difficulty': 'intermediate'
                    })
                if skill_levels.get('skill_machine learning', 0) < 60:
                    recommended_courses.append('Machine Learning Basics')
                    skill_gaps.append({
                        'skill': 'machine_learning',
                        'current_level': skill_levels.get('skill_machine learning', 0),
                        'difficulty': 'advanced'
                    })
            
            elif career_goal == 'Web Development':
                if skill_levels.get('skill_html_css', 0) < 60:
                    recommended_courses.append('HTML and CSS Fundamentals')
                    skill_gaps.append({
                        'skill': 'html_css',
                        'current_level': skill_levels.get('skill_html_css', 0),
                        'difficulty': 'beginner'
                    })
                if skill_levels.get('skill_javascript', 0) < 60:
                    recommended_courses.append('JavaScript Essentials')
                    skill_gaps.append({
                        'skill': 'javascript',
                        'current_level': skill_levels.get('skill_javascript', 0),
                        'difficulty': 'intermediate'
                    })
                if skill_levels.get('skill_react', 0) < 60:
                    recommended_courses.append('React.js Development')
                    skill_gaps.append({
                        'skill': 'react',
                        'current_level': skill_levels.get('skill_react', 0),
                        'difficulty': 'advanced'
                    })

            return {
                'predicted_success_rate': float(predicted_success),
                'recommended_courses': recommended_courses[:3],
                'skill_gaps': skill_gaps
            }

        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            raise

# Initialize the recommender
recommender = CourseRecommender()

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Load the dataset
        df = pd.read_csv('adaptive_learning_dataset.csv')
        
        # Train the model
        success = recommender.train_model(df)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Model trained successfully'
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to train model'
            }), 500

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    try:
        user_data = request.json
        
        # Validate required fields
        required_fields = [
            'age', 'education_level', 'career_goal', 'preferred_learning_style',
            'time_availability_hours_per_week', 'learning_pace', 'weekly_study_hours',
            'platform_visits_per_week', 'engagement_score'
        ]
        
        for field in required_fields:
            if field not in user_data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400

        # Add empty course history if not provided
        if 'course_history' not in user_data:
            user_data['course_history'] = []

        # Get recommendations
        recommendations = recommender.recommend_courses(user_data)
        
        return jsonify({
            'status': 'success',
            'data': recommendations
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    if recommender.feature_importance is None:
        return jsonify({
            'status': 'error',
            'message': 'Model has not been trained yet'
        }), 400
    
    return jsonify({
        'status': 'success',
        'data': recommender.feature_importance.to_dict('records')
    }), 200

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Vercel, 
    # the Vercel server will look for the Flask app directly
    app.run(host='0.0.0.0', debug=False)

# Add this line at the end of the file to expose the Flask app to Vercel
app = app 
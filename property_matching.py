import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PropertyMatcher:
    def __init__(self):
        self.properties_df = None
        self.users_df = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.property_vectors = None
        
    def load_data(self, file_path='Case_Study_2_Data.xlsx'):
        try:
            self.properties_df = pd.read_excel(file_path, sheet_name='Properties')
            self.users_df = pd.read_excel(file_path, sheet_name='Users')
            
            # Convert preferred_locations from string to list
            self.users_df['preferred_locations'] = self.users_df['preferred_locations'].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            )
            
            # Create text vectors for property descriptions
            self.property_vectors = self.vectorizer.fit_transform(self.properties_df['description'])
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def normalize_budget(self, budget):
        if isinstance(budget, str):
            # Remove currency symbols and commas
            budget = budget.replace('$', '').replace(',', '').strip()
            try:
                return float(budget)
            except ValueError:
                return 0
        return float(budget)

    def calculate_match_score(self, property_data, user_preferences):
        scores = {}
        
        # Budget match (30%)
        budget_score = 0
        if user_preferences['budget_min'] <= property_data['price'] <= user_preferences['budget_max']:
            budget_score = 100
        elif property_data['price'] < user_preferences['budget_min']:
            budget_score = 80
        elif property_data['price'] > user_preferences['budget_max']:
            budget_score = 40
        scores['budget_score'] = budget_score * 0.3
        
        # Location match (25%)
        location_score = 0
        if property_data['location'] in user_preferences['preferred_locations']:
            location_score = 100
        scores['location_score'] = location_score * 0.25
        
        # Property type match (20%)
        property_type_score = 0
        if user_preferences['preferred_property_type'] in property_data['description'].lower():
            property_type_score = 100
        scores['property_type_score'] = property_type_score * 0.2
        
        # Features match (15%)
        features_score = 0
        if property_data['bedrooms'] >= user_preferences['min_bedrooms']:
            features_score += 50
        if property_data['bathrooms'] >= user_preferences['min_bathrooms']:
            features_score += 50
        scores['features_score'] = features_score * 0.15
        
        # Description match (10%)
        property_vector = self.vectorizer.transform([property_data['description']])
        similarity = cosine_similarity(property_vector, self.property_vectors).flatten()
        description_score = float(similarity.max() * 100)
        scores['description_score'] = description_score * 0.1
        
        # Calculate total score
        total_score = sum(scores.values())
        scores['total_score'] = total_score
        
        return scores

    def get_top_matches(self, user_id, top_n=5):
        if self.properties_df is None or self.users_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        user_preferences = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
        
        matches = []
        for _, property_data in self.properties_df.iterrows():
            scores = self.calculate_match_score(property_data, user_preferences)
            match_data = {
                'property_id': property_data['property_id'],
                'price': property_data['price'],
                'location': property_data['location'],
                'description': property_data['description'],
                'match_score': round(scores['total_score'], 2),
                'budget_score': round(scores['budget_score'] / 0.3, 2),
                'location_score': round(scores['location_score'] / 0.25, 2),
                'property_type_score': round(scores['property_type_score'] / 0.2, 2),
                'features_score': round(scores['features_score'] / 0.15, 2),
                'description_score': round(scores['description_score'] / 0.1, 2)
            }
            matches.append(match_data)
        
        # Sort by match score and return top N
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        return matches[:top_n] 
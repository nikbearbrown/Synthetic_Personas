#!/usr/bin/env python3
"""
Brewery Survey Simulator

This script simulates survey responses for a craft brewery customer survey
based on customer segments and personality models (OCEAN scores).

Usage:
    python brewery_survey_simulator.py <number_of_respondents>

Example:
    python brewery_survey_simulator.py 500
"""

import sys
import random
import pandas as pd
import numpy as np
from collections import Counter
import json

# Check command line arguments
if len(sys.argv) != 2:
    print("Usage: python brewery_survey_simulator.py <number_of_respondents>")
    sys.exit(1)

try:
    num_respondents = int(sys.argv[1])
    if num_respondents <= 0:
        raise ValueError("Number of respondents must be positive")
except ValueError as e:
    print(f"Error: {e}")
    print("Usage: python brewery_survey_simulator.py <number_of_respondents>")
    sys.exit(1)

# Define customer segments with distributions
SEGMENTS = {
    "Affluent Locals": 0.20,       # 20% of customers
    "Retirees": 0.15,              # 15% of customers
    "Young Professionals": 0.10,   # 10% of customers
    "Summer Tourists": 0.30,       # 30% of customers
    "Day Trippers": 0.15,          # 15% of customers
    "Craft Beer Enthusiasts": 0.07, # 7% of customers
    "Event Attendees": 0.03        # 3% of customers
}

# Define OCEAN scores for each segment (scale of 1-10)
PERSONALITY_MODELS = {
    "Affluent Locals": {
        "Openness": (7, 8),         # (min, max) range
        "Conscientiousness": (7, 8),
        "Extraversion": (6, 7),
        "Agreeableness": (5, 6),
        "Neuroticism": (3, 5)
    },
    "Retirees": {
        "Openness": (5, 6),
        "Conscientiousness": (7, 8),
        "Extraversion": (5, 6),
        "Agreeableness": (7, 8),
        "Neuroticism": (3, 5)
    },
    "Young Professionals": {
        "Openness": (8, 9),
        "Conscientiousness": (6, 7),
        "Extraversion": (7, 8),
        "Agreeableness": (5, 6),
        "Neuroticism": (5, 6)
    },
    "Summer Tourists": {
        "Openness": (7, 8),
        "Conscientiousness": (5, 6),
        "Extraversion": (7, 8),
        "Agreeableness": (6, 7),
        "Neuroticism": (3, 4)
    },
    "Day Trippers": {
        "Openness": (6, 7),
        "Conscientiousness": (5, 6),
        "Extraversion": (6, 7),
        "Agreeableness": (5, 6),
        "Neuroticism": (4, 5)
    },
    "Craft Beer Enthusiasts": {
        "Openness": (8, 9),
        "Conscientiousness": (7, 8),
        "Extraversion": (6, 7),
        "Agreeableness": (4, 6),
        "Neuroticism": (2, 4)
    },
    "Event Attendees": {
        "Openness": (4, 6),
        "Conscientiousness": (6, 7),
        "Extraversion": (7, 9),
        "Agreeableness": (7, 8),
        "Neuroticism": (4, 6)
    }
}

# Age ranges for different segments
AGE_RANGES = {
    "Affluent Locals": (30, 65),
    "Retirees": (65, 85),
    "Young Professionals": (25, 40),
    "Summer Tourists": (21, 65),
    "Day Trippers": (21, 65),
    "Craft Beer Enthusiasts": (25, 55),
    "Event Attendees": (21, 45)
}

# Gender distributions for different segments (Male, Female, Non-binary, Prefer not to say)
GENDER_DIST = {
    "Affluent Locals": [0.48, 0.48, 0.02, 0.02],
    "Retirees": [0.45, 0.53, 0.01, 0.01],
    "Young Professionals": [0.50, 0.44, 0.04, 0.02],
    "Summer Tourists": [0.48, 0.49, 0.02, 0.01],
    "Day Trippers": [0.48, 0.48, 0.03, 0.01],
    "Craft Beer Enthusiasts": [0.60, 0.35, 0.03, 0.02],
    "Event Attendees": [0.45, 0.50, 0.03, 0.02]
}

# Visit frequency distributions for different segments
VISIT_FREQ_DIST = {
    "Affluent Locals": [0.10, 0.60, 0.30, 0],  # Yearly, Monthly, Weekly, Never
    "Retirees": [0.25, 0.65, 0.10, 0],
    "Young Professionals": [0.05, 0.55, 0.40, 0],
    "Summer Tourists": [0.70, 0.25, 0.05, 0],
    "Day Trippers": [0.60, 0.35, 0.05, 0],
    "Craft Beer Enthusiasts": [0.05, 0.30, 0.65, 0],
    "Event Attendees": [0.40, 0.55, 0.05, 0]
}

# Location distributions for different segments
LOCATION_DIST = {
    "Affluent Locals": [0.30, 0.55, 0.15],  # <10min, 10-30min, >30min
    "Retirees": [0.25, 0.60, 0.15],
    "Young Professionals": [0.35, 0.50, 0.15],
    "Summer Tourists": [0.15, 0.40, 0.45],
    "Day Trippers": [0.10, 0.30, 0.60],
    "Craft Beer Enthusiasts": [0.25, 0.50, 0.25],
    "Event Attendees": [0.20, 0.45, 0.35]
}

def weighted_choice(options, weights):
    """Helper function for weighted random selection"""
    return random.choices(options, weights=weights, k=1)[0]

def generate_response(segment, question_type, ocean_scores):
    """Generate response based on OCEAN scores and question type"""
    
    if question_type == "personality_traits":
        # Q6 - personality characteristics (1-5 scale)
        responses = []
        
        # You are a social person - linked to extraversion
        social_score = min(5, max(1, round(ocean_scores["Extraversion"] / 2)))
        responses.append(social_score)
        
        # You live an active lifestyle - linked to conscientiousness and extraversion
        active_score = min(5, max(1, round((ocean_scores["Conscientiousness"] + ocean_scores["Extraversion"]) / 4)))
        responses.append(active_score)
        
        # Alcoholic drink quality is important to you - linked to openness and conscientiousness
        quality_score = min(5, max(1, round((ocean_scores["Openness"] + ocean_scores["Conscientiousness"]) / 4)))
        responses.append(quality_score)
        
        # You like finding new things to do - linked to openness
        novelty_score = min(5, max(1, round(ocean_scores["Openness"] / 2)))
        responses.append(novelty_score)
        
        return responses
    
    elif question_type == "taproom_features":
        # RP1a - Taproom features (multi-select)
        features = []
        
        # Pool table, Card games, Karaoke, Darts - linked to openness and extraversion
        if ocean_scores["Extraversion"] > 6 or ocean_scores["Openness"] > 6:
            for feature in ["Pool table", "Card games", "Karaoke available", "Darts"]:
                if random.random() < (ocean_scores["Extraversion"] + ocean_scores["Openness"]) / 20:
                    features.append(feature)
        
        # Friendly staff - everyone values this somewhat
        if random.random() < 0.7:
            features.append("Friendly staff")
            
        # Easy to socialize - linked to extraversion
        if random.random() < ocean_scores["Extraversion"] / 10:
            features.append("Easy to socialize with other customers")
            
        # Loyalty program - linked to conscientiousness
        if random.random() < ocean_scores["Conscientiousness"] / 10:
            features.append("Loyalty program")
            
        # Easy to network - linked to extraversion and conscientiousness
        if random.random() < (ocean_scores["Extraversion"] + ocean_scores["Conscientiousness"]) / 20:
            features.append("Easy to network with other professionals")
            
        return features
    
    elif question_type == "taproom_factors_ranking":
        # RP1b - Factors ranking (1-8)
        factors = [
            "Drink selection", 
            "Price", 
            "Location", 
            "Casual conversations with employees",
            "Social interactions with other customers",
            "Ability to do other things aside from drink",
            "Networking with other professionals",
            "Earning loyalty rewards"
        ]
        
        # Weight factors based on personality
        weights = {
            "Drink selection": 5 + ocean_scores["Openness"] / 2,  # Higher openness values variety
            "Price": 8 - ocean_scores["Conscientiousness"] / 2,   # Lower conscientiousness more price sensitive
            "Location": 7 - ocean_scores["Openness"] / 2,         # Lower openness values convenience
            "Casual conversations with employees": ocean_scores["Extraversion"] / 2 + ocean_scores["Agreeableness"] / 2,
            "Social interactions with other customers": ocean_scores["Extraversion"],
            "Ability to do other things aside from drink": ocean_scores["Openness"] / 2 + ocean_scores["Extraversion"] / 2,
            "Networking with other professionals": (ocean_scores["Extraversion"] + ocean_scores["Conscientiousness"]) / 3,
            "Earning loyalty rewards": ocean_scores["Conscientiousness"] / 2
        }
        
        # Add some randomness
        for k in weights:
            weights[k] += random.uniform(-1, 1)
            
        # Sort by weights (descending) and convert to ranks
        sorted_factors = sorted(factors, key=lambda x: weights[x], reverse=True)
        ranks = {factor: rank+1 for rank, factor in enumerate(sorted_factors)}
        
        return ranks
    
    elif question_type == "hesitations":
        # RP1c - Hesitations about new taprooms (multi-select)
        hesitations = []
        
        # Not having someone to go with - linked to extraversion (inverse)
        if random.random() < (10 - ocean_scores["Extraversion"]) / 10:
            hesitations.append("Not having someone to go with")
            
        # Difficult to talk to others - linked to extraversion (inverse)
        if random.random() < (10 - ocean_scores["Extraversion"]) / 10:
            hesitations.append("Difficult to talk to others in the taproom")
            
        # Poor customer service - linked to agreeableness
        if random.random() < ocean_scores["Agreeableness"] / 10:
            hesitations.append("Poor customer service")
            
        # No rewards program - linked to conscientiousness
        if random.random() < ocean_scores["Conscientiousness"] / 15:
            hesitations.append("No rewards program")
            
        # No music - linked to openness
        if random.random() < ocean_scores["Openness"] / 15:
            hesitations.append("No music")
            
        # No food - more common hesitation
        if random.random() < 0.4:
            hesitations.append("No food")
            
        return hesitations
    
    elif question_type == "interaction_factors":
        # RP1d - Interaction factors (multi-select)
        factors = []
        
        # Being introduced - linked to extraversion (inverse)
        if random.random() < (10 - ocean_scores["Extraversion"]) / 10:
            factors.append("Being introduced by someone else")
            
        # Good music - linked to openness
        if random.random() < ocean_scores["Openness"] / 10:
            factors.append("Good music playing")
            
        # Joining a game - linked to extraversion and openness
        if random.random() < (ocean_scores["Extraversion"] + ocean_scores["Openness"]) / 20:
            factors.append("Joining a game")
            
        # Trivia - linked to conscientiousness and openness
        if random.random() < (ocean_scores["Conscientiousness"] + ocean_scores["Openness"]) / 20:
            factors.append("Playing competitive trivia")
            
        # Friendliness of staff - everyone values this somewhat
        if random.random() < 0.7:
            factors.append("Friendliness of staff")
            
        # Group activities - linked to extraversion
        if random.random() < ocean_scores["Extraversion"] / 10:
            factors.append("Group activities to do with others present")
            
        return factors
    
    elif question_type == "ideal_features":
        # RP1e - Ideal features (multi-select)
        features = []
        
        # Friendliness of staff - everyone values this somewhat
        if random.random() < 0.7:
            features.append("Friendliness of staff")
            
        # Games - linked to extraversion and openness
        if random.random() < (ocean_scores["Extraversion"] + ocean_scores["Openness"]) / 20:
            features.append("Games available")
            
        # Easy to socialize - linked to extraversion
        if random.random() < ocean_scores["Extraversion"] / 10:
            features.append("Easy to socialize with others")
            
        # Food options - common preference
        if random.random() < 0.5:
            features.append("Variety of food options")
            
        # Outdoor seating - linked to openness
        if random.random() < ocean_scores["Openness"] / 10:
            features.append("Outdoor seating")
            
        # Live music - linked to openness
        if random.random() < ocean_scores["Openness"] / 10:
            features.append("Live music")
            
        # Happy hour - linked to conscientiousness
        if random.random() < ocean_scores["Conscientiousness"] / 10:
            features.append("Happy hour specials")
            
        # TV for sports - varies by segment
        if segment in ["Affluent Locals", "Young Professionals"] and random.random() < 0.4:
            features.append("Large TV screens for sports games")
            
        return features
    
    elif question_type == "primary_reason":
        # Q2a - Primary reason (single select)
        reasons = [
            "To try new and unique craft beers",
            "To socialize with friends/family",
            "To meet new people",
            "To relax and unwind",
            "To attend special events",
            "To enjoy a familiar and consistent experience"
        ]
        
        weights = {
            "To try new and unique craft beers": ocean_scores["Openness"] * 1.2 if segment == "Craft Beer Enthusiasts" else ocean_scores["Openness"],
            "To socialize with friends/family": ocean_scores["Extraversion"] * 0.8 + ocean_scores["Agreeableness"] * 0.2,
            "To meet new people": ocean_scores["Extraversion"] * 0.7 if segment != "Retirees" else ocean_scores["Extraversion"] * 0.3,
            "To relax and unwind": (10 - ocean_scores["Neuroticism"]) * 0.5 + ocean_scores["Agreeableness"] * 0.2,
            "To attend special events": ocean_scores["Openness"] * 0.8 if segment == "Event Attendees" else ocean_scores["Openness"] * 0.4,
            "To enjoy a familiar and consistent experience": (10 - ocean_scores["Openness"]) * 0.7
        }
        
        # Add some randomness
        for k in weights:
            weights[k] += random.uniform(-1, 1)
            
        # Choose based on weights
        return weighted_choice(reasons, [weights[r] for r in reasons])
    
    elif question_type == "consistency_importance":
        # Q2b - Consistency importance (single select)
        options = [
            "Important – I prefer a familiar experience",
            "Somewhat important – I like a balance of new and familiar elements",
            "Not important – I prefer a unique experience each time"
        ]
        
        # Openness is inversely related to desire for consistency
        weights = [
            (10 - ocean_scores["Openness"]) * 0.8,  # Familiar
            5,  # Balance (baseline option)
            ocean_scores["Openness"] * 0.7  # Unique
        ]
        
        return weighted_choice(options, weights)
    
    elif question_type == "experience_likelihood":
        # Q2c - Experience likelihood (1-5 scale for each)
        likelihoods = []
        
        # Themed events - linked to openness
        theme_score = min(5, max(1, round((ocean_scores["Openness"] + random.uniform(-1, 1)) / 2)))
        likelihoods.append(theme_score)
        
        # Social activities - linked to extraversion
        social_score = min(5, max(1, round((ocean_scores["Extraversion"] + random.uniform(-1, 1)) / 2)))
        likelihoods.append(social_score)
        
        # Community events - linked to agreeableness
        community_score = min(5, max(1, round((ocean_scores["Agreeableness"] + random.uniform(-1, 1)) / 2)))
        likelihoods.append(community_score)
        
        # Interactive experiences - linked to openness and extraversion
        interactive_score = min(5, max(1, round((ocean_scores["Openness"] + ocean_scores["Extraversion"] + random.uniform(-1, 1)) / 4)))
        likelihoods.append(interactive_score)
        
        # Unique events - linked to openness
        unique_score = min(5, max(1, round((ocean_scores["Openness"] + random.uniform(-1, 1)) / 2)))
        likelihoods.append(unique_score)
        
        # Consistent environment - inversely linked to openness
        consistent_score = min(5, max(1, round(((10 - ocean_scores["Openness"]) + random.uniform(-1, 1)) / 2)))
        likelihoods.append(consistent_score)
        
        return likelihoods
    
    elif question_type == "community_focus":
        # Q2d - Community focus (single select)
        options = [
            "Yes, I value community-focused experiences",
            "Maybe, if the event aligns with my interests",
            "No, I attend for other reasons"
        ]
        
        # Agreeableness correlates with community focus
        weights = [
            ocean_scores["Agreeableness"] * 0.8,  # Yes
            5,  # Maybe (baseline)
            (10 - ocean_scores["Agreeableness"]) * 0.5  # No
        ]
        
        return weighted_choice(options, weights)
    
    elif question_type == "live_music_interest":
        # Q3a - Live music interest (single select)
        options = [
            "Yes; I have attended before",
            "Yes, I haven't attended yet but I'm interested.",
            "No, I'm not interested."
        ]
        
        # Openness and extraversion correlate with interest in live music
        music_interest = (ocean_scores["Openness"] + ocean_scores["Extraversion"]) / 2
        
        if segment == "Event Attendees":
            weights = [0.6, 0.3, 0.1]  # Event attendees most likely to have gone
        else:
            weights = [
                music_interest / 20,  # Yes, attended
                music_interest / 15,  # Yes, interested
                1 - (music_interest / 20 + music_interest / 15)  # No
            ]
            
        return weighted_choice(options, weights)
    
    elif question_type == "live_music_rating":
        # Q3b - Live music rating (single select)
        options = ["Excellent", "Good", "Neutral", "Poor", "Very Poor"]
        
        # Agreeableness correlates with positive ratings
        weights = [
            ocean_scores["Agreeableness"] / 15,  # Excellent
            ocean_scores["Agreeableness"] / 10,  # Good
            0.3,  # Neutral (baseline)
            (10 - ocean_scores["Agreeableness"]) / 20,  # Poor
            (10 - ocean_scores["Agreeableness"]) / 30  # Very Poor
        ]
        
        return weighted_choice(options, weights)
    
    elif question_type == "music_elements":
        # Q3c - Positive elements of music events (multi-select)
        elements = [
            "Atmosphere and venue ambiance",
            "Quality of live music performances",
            "Variety of performers",
            "High quality of drinks",
            "High quality of food",
            "Social experience and opportunity to meet new people",
            "Special promotions or discounts during the event",
            "Convenience of location",
            "Convenience of timing",
            "Being a fan of the band or specific artists performing"
        ]
        
        # Select 2-5 elements based on personality
        num_selections = random.randint(2, 5)
        
        # Weight elements based on personality
        weights = {
            "Atmosphere and venue ambiance": ocean_scores["Openness"] / 10,
            "Quality of live music performances": ocean_scores["Openness"] / 10,
            "Variety of performers": ocean_scores["Openness"] / 10,
            "High quality of drinks": ocean_scores["Conscientiousness"] / 10,
            "High quality of food": ocean_scores["Conscientiousness"] / 10,
            "Social experience and opportunity to meet new people": ocean_scores["Extraversion"] / 10,
            "Special promotions or discounts during the event": ocean_scores["Conscientiousness"] / 10,
            "Convenience of location": (10 - ocean_scores["Openness"]) / 10,
            "Convenience of timing": ocean_scores["Conscientiousness"] / 10,
            "Being a fan of the band or specific artists performing": ocean_scores["Openness"] / 10
        }
        
        # Normalize weights
        total = sum(weights.values())
        norm_weights = [weights[e]/total for e in elements]
        
        return random.choices(elements, weights=norm_weights, k=num_selections)
    
    elif question_type == "negative_factors":
        # Q3d - Negative factors of music events (multi-select)
        factors = [
            "Uncomfortable venue atmosphere",
            "Poor sound quality",
            "Unappealing performances",
            "Long wait times for orders",
            "Limited drink options",
            "Limited food options",
            "High cover charge",
            "Inconvenient location",
            "Inconvenient timing",
            "Poor event organization"
        ]
        
        # Select 1-3 factors
        num_selections = random.randint(1, 3)
        return random.sample(factors, num_selections)
    
    elif question_type == "preventing_reasons":
        # Q3e - Reasons preventing attendance (multi-select)
        reasons = [
            "Scheduling conflicts",
            "Lack of interest",
            "Preference for a quieter atmosphere at breweries",
            "Unfamiliarity with the performers or event lineup",
            "The cover charge is too expensive",
            "Limited transportation",
            "Inconvenient location",
            "Poor past experiences with similar events"
        ]
        
        # Select 1-3 reasons
        num_selections = random.randint(1, 3)
        return random.sample(reasons, num_selections)
    
    elif question_type == "snack_importance":
        # Q4a - Snack importance (single select)
        options = [
            "Extremely Important",
            "Very Important",
            "Moderately Important",
            "Slightly important",
            "Not at all important"
        ]
        
        # Base likelihood varies by segment
        base_weights = {
            "Affluent Locals": [0.10, 0.25, 0.35, 0.20, 0.10],
            "Retirees": [0.05, 0.20, 0.40, 0.25, 0.10],
            "Young Professionals": [0.15, 0.30, 0.35, 0.15, 0.05],
            "Summer Tourists": [0.20, 0.30, 0.30, 0.15, 0.05],
            "Day Trippers": [0.20, 0.35, 0.30, 0.10, 0.05],
            "Craft Beer Enthusiasts": [0.05, 0.15, 0.30, 0.30, 0.20],
            "Event Attendees": [0.15, 0.30, 0.35, 0.15, 0.05]
        }
        
        return weighted_choice(options, base_weights[segment])
    
    elif question_type == "ai_trust":
        # Q5b - AI trust (single select)
        options = [
            "Not at all",
            "A little",
            "A moderate amount",
            "A lot",
            "A great deal"
        ]
        
        # Openness correlates with trust in new technologies
        weights = [
            (10 - ocean_scores["Openness"]) / 10,  # Not at all
            (10 - ocean_scores["Openness"]) / 15,  # A little
            0.3,  # Moderate (baseline)
            ocean_scores["Openness"] / 15,  # A lot
            ocean_scores["Openness"] / 20  # Great deal
        ]
        
        return weighted_choice(options, weights)
    
    else:
        # For other questions, provide reasonable default based on segment
        return None

# Function to generate a respondent's complete survey data
def generate_respondent(segment):
    """Generate a complete survey response for a respondent of the given segment"""
    
    # Get OCEAN scores for this segment
    ocean_scores = {}
    for trait, (min_val, max_val) in PERSONALITY_MODELS[segment].items():
        ocean_scores[trait] = random.uniform(min_val, max_val)
    
    # Set up the respondent data
    respondent = {}
    
    # Q1 - Gender
    respondent["Gender"] = weighted_choice(
        ["Male", "Female", "Non-binary or other gender", "Prefer not to say"],
        GENDER_DIST[segment]
    )
    
    # Q2 - Above 21
    respondent["Above 21"] = "Yes"  # All respondents are 21+
    
    # Q3 - Age
    respondent["Age"] = random.randint(*AGE_RANGES[segment])
    
    # Q4 - Visit frequency
    respondent["Visit Frequency"] = weighted_choice(
        ["At least once a year", "At least once a month", "At least once a week", "Never"],
        VISIT_FREQ_DIST[segment]
    )
    
    # Q5 - Location
    respondent["Location"] = weighted_choice(
        ["Less than 10 minutes away", "Between 10 and 30 minutes", "More than 30 minutes away"],
        LOCATION_DIST[segment]
    )
    
    # Q6 - Personality traits
    personality_scores = generate_response(segment, "personality_traits", ocean_scores)
    traits = ["Social person", "Active lifestyle", "Drink quality important", "Like finding new things"]
    for i, trait in enumerate(traits):
        respondent[f"Trait: {trait}"] = personality_scores[i]
    
    # RP1a - Taproom features
    respondent["Taproom Features"] = generate_response(segment, "taproom_features", ocean_scores)
    
    # RP1b - Factors ranking
    factor_ranks = generate_response(segment, "taproom_factors_ranking", ocean_scores)
    for factor, rank in factor_ranks.items():
        respondent[f"Rank: {factor}"] = rank
    
    # RP1c - Hesitations
    respondent["Hesitations"] = generate_response(segment, "hesitations", ocean_scores)
    
    # RP1d - Interaction factors
    respondent["Interaction Factors"] = generate_response(segment, "interaction_factors", ocean_scores)
    
    # RP1e - Ideal features
    respondent["Ideal Features"] = generate_response(segment, "ideal_features", ocean_scores)
    
    # Q2a - Primary reason
    respondent["Primary Reason"] = generate_response(segment, "primary_reason", ocean_scores)
    
    # Q2b - Consistency importance
    respondent["Consistency Importance"] = generate_response(segment, "consistency_importance", ocean_scores)
    
    # Q2c - Experience likelihood
    experience_scores = generate_response(segment, "experience_likelihood", ocean_scores)
    experiences = ["Themed events", "Social activities", "Community events", 
                   "Interactive experiences", "Unique events", "Consistent environment"]
    for i, exp in enumerate(experiences):
        respondent[f"Experience: {exp}"] = experience_scores[i]
    
    # Q2d - Community focus
    respondent["Community Focus"] = generate_response(segment, "community_focus", ocean_scores)
    
    # Q2e - Return frequency factor - text field, generate from templates
    return_factors = [
        "Offer a wider selection of craft beers",
        "Have more frequent events or special nights",
        "Provide comfortable seating and atmosphere",
        "Improve the quality of food offerings",
        "Offer loyalty rewards for frequent customers",
        "Host themed nights or special tastings",
        "Have friendlier staff and better service",
        "Offer more unique or experimental brews",
        "Have live music more often",
        "Create a quieter environment for conversation",
        "Add more games or activities",
        "Extend happy hour specials"
    ]
    respondent["Return Factor"] = random.choice(return_factors)
    
    # Q3a - Live music interest
    respondent["Live Music Interest"] = generate_response(segment, "live_music_interest", ocean_scores)
    
    # Q3b to Q3e - Live music questions - only fill relevant ones
    if respondent["Live Music Interest"] == "Yes; I have attended before":
        respondent["Live Music Rating"] = generate_response(segment, "live_music_rating", ocean_scores)
        
        if respondent["Live Music Rating"] in ["Excellent", "Good"]:
            # Q3c - Positive elements
            respondent["Positive Music Elements"] = generate_response(segment, "music_elements", ocean_scores)
        elif respondent["Live Music Rating"] in ["Poor", "Very Poor"]:
            # Q3d - Negative factors
            respondent["Negative Music Factors"] = generate_response(segment, "negative_factors", ocean_scores)
    elif respondent["Live Music Interest"] == "Yes, I haven't attended yet but I'm interested.":
        # Q3c - Positive elements
        respondent["Positive Music Elements"] = generate_response(segment, "music_elements", ocean_scores)
    elif respondent["Live Music Interest"] == "No, I'm not interested.":
        # Q3e - Reasons preventing attendance
        respondent["Preventing Reasons"] = generate_response(segment, "preventing_reasons", ocean_scores)
    
    # Q4a - Snack importance
    respondent["Snack Importance"] = generate_response(segment, "snack_importance", ocean_scores)
    
    # Q4b - Snack order frequency
    snack_freq_options = ["Everytime", "Most of the time", "Sometimes", "Rarely", "Never"]
    
    # Base probabilities adjusted by segment
    if segment == "Craft Beer Enthusiasts":
        snack_weights = [0.05, 0.15, 0.30, 0.30, 0.20]  # Less likely to order snacks
    elif segment in ["Summer Tourists", "Day Trippers"]:
        snack_weights = [0.15, 0.30, 0.35, 0.15, 0.05]  # More likely to order snacks
    else:
        snack_weights = [0.10, 0.25, 0.40, 0.20, 0.05]  # Average
        
    respondent["Snack Order Frequency"] = weighted_choice(snack_freq_options, snack_weights)
    
    # Q4c - Snack stay longer
    stay_options = ["Very Likely", "Likely", "Neutral", "Unlikely", "Very Unlikely"]
    
    # Determine likelihood based on prior answers
    if respondent["Snack Order Frequency"] in ["Everytime", "Most of the time"]:
        stay_weights = [0.40, 0.40, 0.15, 0.03, 0.02]
    elif respondent["Snack Order Frequency"] == "Sometimes":
        stay_weights = [0.20, 0.35, 0.30, 0.10, 0.05]
    else:
        stay_weights = [0.05, 0.15, 0.30, 0.30, 0.20]
        
    respondent["Snack Stay Longer"] = weighted_choice(stay_options, stay_weights)
    
    # Q4d - Snack type preference
    snack_types = [
        "Pre-packaged Snacks",
        "Warm-prepared Snacks",
        "Sweets",
        "Cold Bites"
    ]
    
    # Number of selections varies by segment
    if segment in ["Affluent Locals", "Retirees"]:
        num_selections = weighted_choice([1, 2, 3, 4], [0.2, 0.4, 0.3, 0.1])
    else:
        num_selections = weighted_choice([1, 2, 3, 4], [0.3, 0.4, 0.2, 0.1])
        
    respondent["Snack Type Preference"] = random.sample(snack_types, num_selections)
    
    # Q4e - Pairing interest
    pairing_options = ["Yes", "Maybe", "No"]
    
    if segment == "Craft Beer Enthusiasts":
        pairing_weights = [0.60, 0.30, 0.10]  # Most interested in pairings
    elif segment in ["Young Professionals", "Affluent Locals"]:
        pairing_weights = [0.40, 0.45, 0.15]  # Moderately interested
    else:
        pairing_weights = [0.30, 0.50, 0.20]  # Average interest
        
    respondent["Pairing Interest"] = weighted_choice(pairing_options, pairing_weights)
    
    # Q5a - AI consultation
    ai_options = ["Yes", "No"]
    
    # Adjust by age and openness
    if respondent["Age"] < 40 and ocean_scores["Openness"] > 6:
        ai_prob = 0.40  # Younger, open people more likely to use AI
    elif respondent["Age"] > 65:
        ai_prob = 0.15  # Older less likely
    else:
        ai_prob = 0.25  # Average
        
    respondent["AI Consultation"] = weighted_choice(ai_options, [ai_prob, 1-ai_prob])
    
    # Q5b - AI trust
    respondent["AI Trust"] = generate_response(segment, "ai_trust", ocean_scores)
    
    # Q5c - AI assistant interest
    ai_interest_options = [
        "Not interested",
        "Slightly interested",
        "Moderately interested",
        "Very interested",
        "Extremely interested"
    ]
    
    # Base on previous AI responses and openness
    if respondent["AI Consultation"] == "Yes" and ocean_scores["Openness"] > 6:
        interest_weights = [0.05, 0.15, 0.30, 0.35, 0.15]  # High interest
    elif respondent["AI Consultation"] == "Yes":
        interest_weights = [0.10, 0.20, 0.40, 0.20, 0.10]  # Moderate interest
    elif ocean_scores["Openness"] > 7:
        interest_weights = [0.15, 0.25, 0.35, 0.20, 0.05]  # Open but haven't used
    else:
        interest_weights = [0.30, 0.35, 0.25, 0.08, 0.02]  # Low interest
        
    respondent["AI Assistant Interest"] = weighted_choice(ai_interest_options, interest_weights)
    
    # Q5d - AI satisfaction (only if used)
    if respondent["AI Consultation"] == "Yes":
        satisfaction_options = [
            "Hated it",
            "Very dissatisfied",
            "Somewhat dissatisfied",
            "Neutral",
            "Somewhat satisfied", 
            "Very satisfied",
            "Loved it"
        ]
        
        # Most users are moderately satisfied
        satisfaction_weights = [0.02, 0.05, 0.08, 0.25, 0.35, 0.20, 0.05]
        respondent["AI Satisfaction"] = weighted_choice(satisfaction_options, satisfaction_weights)
    
    # Q5e - AI valuable features - text field, generate from templates
    ai_features = [
        "Personalized beer recommendations based on my taste preferences",
        "Information about upcoming events and special releases",
        "Beer pairing suggestions for food items",
        "Wait time estimates for busy periods",
        "Answers to questions about brewing processes and ingredients",
        "Local transportation options and directions",
        "Information about seasonal or limited edition brews",
        "Ability to reserve tables or event tickets",
        "Updates on specials or happy hours",
        "History and background of the brewery"
    ]
    respondent["AI Valuable Features"] = random.choice(ai_features)
    
    # Add segment info for analysis
    respondent["Segment"] = segment
    
    return respondent

# Generate the requested number of respondents
def generate_survey_data(num_respondents):
    """Generate survey data for the specified number of respondents"""
    respondents = []
    
    # Distribute respondents according to segment distribution
    segment_counts = {}
    remaining = num_respondents
    
    for segment, proportion in list(SEGMENTS.items())[:-1]:  # All except the last segment
        count = int(num_respondents * proportion)
        segment_counts[segment] = count
        remaining -= count
    
    # Assign the remainder to the last segment
    last_segment = list(SEGMENTS.keys())[-1]
    segment_counts[last_segment] = remaining
    
    # Generate respondents for each segment
    for segment, count in segment_counts.items():
        for _ in range(count):
            respondents.append(generate_respondent(segment))
    
    # Shuffle to randomize order
    random.shuffle(respondents)
    
    return respondents

def analyze_survey_data(df):
    """Generate basic analysis of the survey data"""
    
    print("\n=== SURVEY ANALYSIS ===\n")
    
    # 1. Demographics
    print("DEMOGRAPHICS:")
    print(f"- Number of respondents: {len(df)}")
    print(f"- Gender distribution: {df['Gender'].value_counts(normalize=True).apply(lambda x: f'{x:.1%}').to_dict()}")
    print(f"- Average age: {df['Age'].mean():.1f} years")
    
    # 2. Visit patterns by segment
    print("\nVISIT FREQUENCY BY SEGMENT:")
    segment_visit = pd.crosstab(df['Segment'], df['Visit Frequency'], normalize='index')
    for segment in df['Segment'].unique():
        print(f"- {segment}:")
        for freq, pct in segment_visit.loc[segment].items():
            print(f"  - {freq}: {pct:.1%}")
    
    # 3. Primary reasons for visiting
    print("\nPRIMARY REASONS FOR VISITING:")
    reasons = df['Primary Reason'].value_counts(normalize=True)
    for reason, pct in reasons.items():
        print(f"- {reason}: {pct:.1%}")
    
    # 4. Features that encourage return visits
    print("\nTOP FEATURES THAT ENCOURAGE RETURN VISITS:")
    # Flatten the list columns for analysis
    all_features = []
    for features in df["Ideal Features"]:
        all_features.extend(features)
    
    feature_counts = Counter(all_features)
    for feature, count in feature_counts.most_common(5):
        print(f"- {feature}: {count} mentions ({count/len(df):.1%} of respondents)")
    
    # 5. Live music interest
    print("\nLIVE MUSIC INTEREST:")
    music_interest = df['Live Music Interest'].value_counts(normalize=True)
    for option, pct in music_interest.items():
        print(f"- {option}: {pct:.1%}")
    
    # 6. Snack importance and behavior
    print("\nSNACK PREFERENCES:")
    snack_importance = df['Snack Importance'].value_counts(normalize=True)
    print("Importance of snack availability:")
    for level, pct in snack_importance.items():
        print(f"- {level}: {pct:.1%}")
    
    print("\nSnack types preferred:")
    all_snack_types = []
    for types in df["Snack Type Preference"]:
        all_snack_types.extend(types)
    
    snack_type_counts = Counter(all_snack_types)
    for snack_type, count in snack_type_counts.most_common():
        print(f"- {snack_type}: {count} mentions ({count/len(df):.1%} of respondents)")
    
    # 7. AI interest and usage
    print("\nAI USAGE AND INTEREST:")
    ai_consultation = df['AI Consultation'].value_counts(normalize=True)
    print(f"- Have used AI for recommendations: {ai_consultation.get('Yes', 0):.1%}")
    
    ai_interest = df['AI Assistant Interest'].value_counts(normalize=True)
    print("Interest in brewery-specific AI assistant:")
    for level, pct in ai_interest.items():
        print(f"- {level}: {pct:.1%}")

# Main function
def main():
    print(f"Generating survey data for {num_respondents} respondents...")
    
    # Set seed for reproducibility
    random.seed(42)
    
    # Generate the data
    respondents = generate_survey_data(num_respondents)
    
    # Convert to dataframe
    df = pd.DataFrame(respondents)
    
    # Save to CSV
    output_file = "brewery_survey_responses.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Generated {len(respondents)} survey responses")
    print(f"Data saved to {output_file}")
    
    # Display segment distribution
    segment_counts = Counter([r["Segment"] for r in respondents])
    print("\nSegment Distribution:")
    for segment, count in segment_counts.items():
        print(f"{segment}: {count} ({count/num_respondents*100:.1f}%)")
    
    # Run analysis
    analyze_survey_data(df)

if __name__ == "__main__":
    main()
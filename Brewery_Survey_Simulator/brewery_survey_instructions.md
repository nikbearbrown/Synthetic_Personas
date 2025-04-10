# Brewery Survey Simulator for Falmouth, MA

This Python script simulates survey responses for a craft brewery in Falmouth, MA based on the customer segments and personality models you've provided. The simulator creates realistic survey data that reflects the characteristics and preferences of different customer types visiting a craft brewery in this region.

## Features

- Generates survey responses based on the following customer segments:
  - Affluent Locals (20%)
  - Retirees (15%)
  - Young Professionals (10%)
  - Summer Tourists (30%)
  - Day Trippers (15%)
  - Craft Beer Enthusiasts (7%)
  - Event Attendees (3%)

- Each segment has distinct personality traits based on the OCEAN (Big Five) model:
  - Openness to experience
  - Conscientiousness
  - Extraversion
  - Agreeableness
  - Neuroticism

- The survey covers multiple aspects of brewery experiences:
  - Demographics
  - Visit patterns
  - Taproom features and preferences
  - Social interaction factors
  - Live music interest
  - Food and snack preferences
  - AI recommendation experiences

## How to Use

1. Save the `brewery_survey_simulator.py` file to your local machine
2. Open a terminal or command prompt
3. Navigate to the directory containing the script
4. Run the script with the number of respondents as an argument:

```bash
python brewery_survey_simulator.py 500
```

Replace `500` with your desired number of respondents.

## Output

The script will:

1. Generate the specified number of simulated survey responses
2. Save the data to a CSV file named `brewery_survey_responses.csv`
3. Display a basic analysis of the results, including:
   - Demographics breakdown
   - Visit frequency by segment
   - Top reasons for visiting
   - Most popular taproom features
   - Live music interest
   - Snack preferences
   - AI usage and interest

## Customization

You can modify the script to adjust:

- The distribution of customer segments (SEGMENTS dictionary)
- Personality trait ranges for each segment (PERSONALITY_MODELS dictionary)
- Age ranges, gender distribution, and visit frequency patterns
- Response probabilities for different questions

## Requirements

- Python 3.6 or higher
- Required packages:
  - pandas
  - numpy

Install the required packages using:

```bash
pip install pandas numpy
```

## Notes for Falmouth, MA Specific Analysis

The simulator is tailored to the Falmouth, MA context with:

- Emphasis on seasonal population variations (high summer tourism)
- Age distributions aligned with Falmouth's demographic profile (higher percentage of retirees)
- Geographic considerations for "Location" responses (accounting for Falmouth's position as a gateway to Martha's Vineyard)
- Customer segments that reflect the town's year-round and seasonal population dynamics

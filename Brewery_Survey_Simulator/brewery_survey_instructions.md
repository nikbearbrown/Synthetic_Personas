
# Brewery Survey Simulator for Bears Brewery in Barrelford, MA

This Python script simulates survey responses for **Bears Brewery**, a beloved craft brewery located in **Barrelford**, a fictional New England town with deep brewing roots. The simulator generates realistic survey data based on local and seasonal customer segments, reflecting the personality traits and preferences of different visitor types.

---

## About Barrelford, MA

Nestled on a fog-kissed coastline and surrounded by pine-draped hills, **Barrelford** was founded in **1738** by English settlers and German immigrants. The town quickly earned a reputation for its underground brewing scene—literally. During colonial times, locals fermented ales in oak barrels beneath the old granary to dodge both taxes and temperance.

By the early 1800s, the town was known as **“the Brew Basket of the Bay,”** a title it wears proudly even today. That same rebellious, creative spirit is now channeled into **Bears Brewery**, a cornerstone of the community and a magnet for both locals and seasonal adventurers.

---

## Features

This simulator generates survey responses based on realistic customer profiles visiting Bears Brewery:

### Customer Segments

- **Affluent Locals** (20%)  
- **Retirees** (15%)  
- **Young Professionals** (10%)  
- **Summer Tourists** (30%)  
- **Day Trippers** (15%)  
- **Craft Beer Enthusiasts** (7%)  
- **Event Attendees** (3%)  

### Personality Models (OCEAN – Big Five)

Each segment has distinct traits across:

- Openness to experience  
- Conscientiousness  
- Extraversion  
- Agreeableness  
- Neuroticism  

### Survey Topics Covered

- Demographics  
- Visit patterns  
- Taproom feature preferences  
- Social atmosphere and interaction  
- Interest in live music  
- Food and snack preferences  
- Experiences with AI-based recommendations  

---

## How to Use

1. Save the `brewery_survey_simulator.py` file to your local machine.  
2. Open a terminal or command prompt.  
3. Navigate to the directory where the script is saved.  
4. Run the script with your desired number of respondents:

```bash
python brewery_survey_simulator.py 500
```

Replace `500` with any number you want.

---

## Output

The script will:

1. Generate simulated survey responses for the specified number of visitors  
2. Save the data in a CSV file named `brewery_survey_responses.csv`  
3. Print a summary that includes:
   - Demographic breakdowns  
   - Visit frequency by segment  
   - Top reasons for visiting  
   - Most popular taproom features  
   - Interest in live music  
   - Food and snack preferences  
   - AI usage and sentiment  

---

## Customization

You can tweak the simulation logic to better fit your needs:

- Adjust customer segment weights (`SEGMENTS` dictionary)  
- Change personality profiles (`PERSONALITY_MODELS` dictionary)  
- Modify demographics (e.g., age, gender, location)  
- Tune probabilities for preferences and behaviors  

---

## Requirements

- Python **3.6 or higher**  
- Required Python packages:
  - `pandas`
  - `numpy`

Install them using:

```bash
pip install pandas numpy
```

---

## Notes for the Barrelford Context

This simulator is designed to reflect the unique character of **Barrelford, MA**:

- Seasonal population swings due to heavy summer tourism  
- A high proportion of retirees in the year-round population  
- A steady stream of day-trippers from nearby coastal towns  
- Taproom and event preferences inspired by New England craft beer culture  

From secret basement brews in the 1700s to modern AI-powered recommendations, the story of **Bears Brewery** and its community lives on in every simulated response.

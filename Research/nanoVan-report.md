# Factor Analysis and PCA in Marketing: The NanoVan Case Study
*Viet Khue "Kevin" Nguyen & Nik Bear Brown*  
*Monday, April 7, 2025*

## Abstract

This study demonstrates the application of dimension reduction and market segmentation techniques for marketing research using the NanoVan case study. We compare Factor Analysis and Principal Component Analysis (PCA) to uncover underlying patterns in consumer preferences for a novel vehicle concept.

Our analysis employs a comprehensive methodological approach beginning with factor analysis to identify latent consumer dimensions, followed by cluster analysis for market segmentation, and concluding with PCA for methodological comparison. The factor analysis reveals five distinct dimensions driving NanoVan preferences: Luxury Orientation, Compact Versatility, Family Transportation Focus, Environmental Consciousness, and Safety and Performance. Subsequent cluster analysis identifies three natural market segments with varying levels of interest in the concept.

The comparison with PCA highlights important differences between the two techniques, with factor analysis generally producing more interpretable dimensions for marketing strategy development. Regression analysis using both factor scores and principal components demonstrates comparable predictive power for concept liking, with key factors significantly influencing consumer interest.

This case study provides valuable insights for marketers at Lake View Associates by transforming complex survey data into actionable consumer segments and targeting strategies. Our findings suggest focusing marketing efforts on consumers valuing luxury features and compact versatility while addressing potential concerns regarding safety and performance. Beyond the specific case, this analysis offers a methodological template for applying dimensional reduction techniques to marketing research, demonstrating how statistical approaches can support strategic decision-making in new product development.

## INTRODUCTION

As Peter Drucker once said: "Trying to predict the future is like trying to drive down a country road at night with no lights, while looking out the back window." Despite the dark humor of the late Austrian American consultant and educator, it seems to go against the human condition to never partake this comically dangerous endeavor as we can see countless attempts of doing so, especially in many modern businesses. Ironically, the U.S. Auto Industry is also a part of these daredevils who would not be deterred by the darkness of ignorance for the shine of profit is enough to light the way. Fortunately for them, they are not driving alone on this treacherous road; many firms were created for the sole purpose of charting this road, and Lake View Associates (LVA) is one of these who have dedicated themselves to figuring out the best way for the Auto Industry to move forward. Within the context of this case study, it would seem that LVA has discovered a clue, or rather a trend, that could indicate a possibly profitable route for the Auto Industry after they have collected data from their ongoing biannual consumer panel surveys. This report aims to help LVA prepare an analysis for upper management discussing the viability of a new concept, called "NanoVan", and the different profiles of the potential segments that would help understand its potential for success in the US market.

In order to reach the goal stated above, this report is organized into the following sections:

1. Data Verification
2. Establish Relationship with Concept Liking
3. Data Reduction through Factor Analysis
4. Explanation of Factors
5. Market Segmentation
6. "Reality Check" relating Clusters to Demographics
7. Comparison of Factor Analysis with PCA
8. Strategic Recommendations

## 1. Data Verification

The NanoVan dataset contains 400 respondents with 39 variables, including 30 attribute variables measuring consumer preferences, demographic information, and the concept liking score (nvliking). Initial verification confirmed the dataset was complete with no missing values.

[**SUGGESTED VISUALIZATION: Histograms of attribute variables showing distribution patterns**]

Examination of the attribute variables through histograms showed consistent distributions, suggesting standardized measurement (likely 1-9 Likert scales). This uniformity across variables is typical of well-designed survey instruments.

During verification, an ID field ("subjnumb") was identified and removed from analysis as it contained sequential identifiers with no analytical meaning that would distort statistical relationships.

[**SUGGESTED VISUALIZATION: Boxplot of attribute variables to show distributions without the ID field**]

Boxplots revealed no significant outliers that would negatively impact analysis. The correlation matrix showed meaningful patterns of relationships between variables, with visible clusters of positive and negative correlations, confirming the data captured real underlying relationships rather than random noise.

[**SUGGESTED VISUALIZATION: Correlation matrix heatmap showing relationships between variables**]

## 2. Establish Relationship with Concept Liking

A multiple linear regression was conducted to determine how the 30 attribute variables relate to the overall concept liking for the NanoVan ("nvliking"). The regression model was statistically significant (F = 7.314, p < 0.001) and explained 37.29% of the variance in NanoVan liking (R-squared = 0.3729, Adjusted R-squared = 0.322).

Despite the reasonable overall model fit, only two variables showed statistical significance at p < 0.05:
- "shdcarpl" (Should carpool): Negative relationship (-0.287, p=0.020)
- "lthrbetr" (Leather seats better): Positive relationship (0.248, p=0.043)

Several variables approached significance (p < 0.10), including "carefmny," "perfimpt," "suvcmpct," "homlrgst," "strngwrn," and "twoincom." The low number of significant predictors despite a decent R-squared suggested potential multicollinearity among variables, supporting the need for dimension reduction.

[**SUGGESTED VISUALIZATION: Bar chart of regression coefficients with significant variables highlighted**]

The regression revealed that luxury preferences, family transportation needs, and performance orientation positively influenced NanoVan liking, while environmental consciousness and economic frugality correlated with lower interest.

## 3. Data Reduction through Factor Analysis

### a. Appropriateness of Factor Analysis

Factor analysis appropriateness was assessed using two statistical tests:

**Bartlett's Test of Sphericity:**
- Chi-square: 7884.07
- p-value: effectively zero (p < 0.0000000001)
- The highly significant result rejected the null hypothesis that variables are uncorrelated

**Kaiser-Meyer-Olkin (KMO) Test:**
- KMO Score: 0.9233 (classified as "Marvelous")
- This exceptional score indicated excellent sampling adequacy

Both tests provided compelling statistical evidence that the data was ideal for factor analysis. The extremely low p-value from Bartlett's test confirmed significant correlations exist, while the very high KMO score indicated clear correlation patterns that could be effectively reduced to underlying factors.

### b. Determining the Number of Factors

The optimal number of factors was determined using multiple methods:

**Kaiser Criterion (Eigenvalue > 1):**
- Identified 5 factors with eigenvalues greater than 1

[**SUGGESTED VISUALIZATION: Scree plot showing eigenvalues for each potential factor**]

**Scree Plot Analysis:** The scree plot showed eigenvalues for each potential factor:
- Eigenvalues:
  - Factor 1: 8.28
  - Factor 2: 5.01
  - Factor 3: 3.09
  - Factor 4: 2.70
  - Factor 5: 1.80
  - Factor 6: 0.63

A clear "elbow" appeared after the fifth factor, with a substantial drop between factors 5 and 6, supporting the 5-factor solution.

**Variance Explained:** The cumulative variance explained by the first five factors was 69.6%:
- Factor 1: 27.6%
- Factor 2: 16.7%
- Factor 3: 10.3%
- Factor 4: 9.0%
- Factor 5: 6.0%

The sixth factor would only add 2.1% more explained variance, further supporting the 5-factor solution.

[**SUGGESTED VISUALIZATION: Bar chart showing variance explained by each factor**]

### c. Factor Extraction and Interpretation

A 5-factor solution with Varimax rotation was extracted to maximize interpretability. The rotated solution demonstrated clean loading patterns with most variables loading strongly on only one factor, facilitating clear interpretation.

[**SUGGESTED VISUALIZATION: Heatmap of factor loadings with values < 0.4 suppressed for clarity**]

Additionally, we tested a 6-factor solution to verify our decision. The sixth factor showed no significant loadings (all below 0.4), confirming that the 5-factor solution was optimal. The sixth factor contained no meaningful pattern and would only complicate interpretation without adding substantial explained variance.

### d. Factor Naming and Interpretation

Based on the pattern of loadings, the five factors have been named and interpreted as follows:

**Factor 1: Luxury Orientation**
- Strong positive loadings: buyhghnd (0.79), pricqual (0.72), lthrbetr (0.71), prmsound (0.66), stylclth (0.61), tkvacatn (0.67), twoincom (0.68), accesfun (0.68), imprtapp (0.51)
- Strong negative loadings: carefmny (-0.77), passnimp (-0.65)
- Interpretation: This factor represents consumers who prioritize luxury and quality in their vehicle purchases. They believe in spending for quality (price reflects quality), prefer premium features (leather seats), and enjoy vehicle customization (accessories). The negative relationship with being careful with money suggests these consumers are willing to spend more for premium features.

**Factor 2: Compact Versatility**
- Strong positive loadings: miniboxy (0.83), suvcmpct (0.80), noparkrm (0.78), secbiggr (0.72), needbetw (0.71), nohummer (0.66)
- Strong negative loadings: homlrgst (-0.66), next2str (-0.71)
- Interpretation: This factor identifies consumers who want a right-sized vehicle that offers versatility without excess bulk. They find current minivans too large and boxy but need more space than a sedan provides. They have parking constraints and want a "Goldilocks" solution - not too big, not too small.

**Factor 3: Family Transportation Focus**
- Strong positive loadings: kidtrans (0.96), kidsbulk (0.77), aftrschl (0.71)
- Strong negative loadings: nordtrps (-0.82)
- Interpretation: This factor clearly represents parents who need a vehicle primarily for transporting children and their belongings to various activities. The negative loading on "no road trips" suggests these families also use their vehicles for family travel.

**Factor 4: Environmental Consciousness**
- Strong positive loadings: shdcarpl (0.81), tkvacatn (0.42)
- Strong negative loadings: envrminr (-0.84), wntguzlr (-0.69)
- Interpretation: This factor represents environmentally-conscious consumers. The negative loadings on "environmental impact minor" and "will buy gas guzzler" combined with the positive loading on carpooling suggest strong environmental values. These consumers reject the idea that environmental impact is minor and oppose gas-guzzling vehicles.

**Factor 5: Safety and Performance**
- Strong positive loadings: safeimpt (0.89), lk4whldr (0.81), strngwrn (0.67)
- Strong negative loadings: perfimpt (-0.85)
- Interpretation: This factor identifies consumers who prioritize safety features, four-wheel drive capability (possibly for security in adverse conditions), and warranty protection. The negative loading on performance importance suggests these consumers prioritize safety and reliability over pure driving performance.

## 4. Explanation of Factor Influence

When the factor scores replaced the original 30 attribute variables in predicting NanoVan liking, the regression model produced an R-squared of 0.337 (compared to 0.373 with all original variables). This represents only a slight decrease in explanatory power despite reducing from 30 variables to just 5 factors (an 83% reduction in complexity).

[**SUGGESTED VISUALIZATION: Bar chart comparing R-squared values for original variables vs. factor scores**]

The factor regression results revealed:

**Significant Factors (p < 0.05):**
- Factor 1 (Luxury Orientation): Strong positive effect (β = 1.09, p < 0.001)
- Factor 2 (Compact Versatility): Strong positive effect (β = 1.03, p < 0.001)
- Factor 5 (Safety and Performance): Strong negative effect (β = -0.58, p < 0.001)

**Non-Significant Factors (p > 0.05):**
- Factor 3 (Family Transportation): Positive but not significant (β = 0.20, p = 0.076)
- Factor 4 (Environmental Consciousness): Negative but marginally significant (β = -0.22, p = 0.062)

[**SUGGESTED VISUALIZATION: Bar chart showing factor coefficients in predicting NanoVan liking with significant factors highlighted**]

This regression suggests that NanoVan appeal is primarily driven by luxury preferences and desire for compact versatility, while concerns about safety and performance negatively impact interest. The slight decrease in R-squared is a reasonable trade-off for the substantial gain in interpretability and strategic insight.

## 5. Market Segmentation

### a. Hierarchical Cluster Analysis

Hierarchical cluster analysis was performed using the factor scores to determine the optimal number of clusters. The dendrogram showed three distinct branches, suggesting three natural groupings of consumers.

[**SUGGESTED VISUALIZATION: Dendrogram showing hierarchical clustering results**]

### b. Silhouette Analysis

To validate the optimal number of clusters, silhouette analysis was conducted. The silhouette score was highest at k=3 (approximately 0.215), confirming that three clusters represented the most cohesive and well-separated groupings.

[**SUGGESTED VISUALIZATION: Line graph showing silhouette scores for different numbers of clusters**]

### c. K-means Cluster Analysis

Based on both the hierarchical clustering and silhouette score analysis, a 3-cluster solution was implemented using K-means clustering. The resulting clusters can be profiled based on their factor score patterns:

[**SUGGESTED VISUALIZATION: Scatter plot showing clusters in 2D factor space (Factors 1 and 2)**]

The three identified market segments are:

**Cluster 0: Value-Conscious Families**
- High on Family Transportation Focus
- Moderately high on Environmental Consciousness
- Low on Luxury Orientation
- Moderate on Compact Versatility
- Moderate on Safety and Performance

This segment consists of family-oriented consumers who need practical transportation for children but are not interested in luxury features.

**Cluster 1: Urban Luxury Seekers**
- Very high on Luxury Orientation
- High on Compact Versatility
- Low on Family Transportation Focus
- Low on Environmental Consciousness
- Low on Safety and Performance

These consumers value premium features and styling but live in urban environments requiring more compact vehicles.

**Cluster 2: Practical Safety-Conscious Consumers**
- Very high on Safety and Performance
- Moderate on Family Transportation Focus
- Low on Luxury Orientation
- Low on Compact Versatility
- High on Environmental Consciousness

This segment prioritizes safety, reliability, and practical performance over luxury or compact sizing.

## 6. Reality Check: Connecting Clusters to Concept Interest and Demographics

### a. Concept Liking by Cluster

Analysis of NanoVan liking scores by cluster revealed significant differences (ANOVA: F = 33.22, p < 0.001):
- Cluster 0 (Value-Conscious Families): Moderate liking (Mean = 5.1)
- Cluster 1 (Urban Luxury Seekers): Highest liking (Mean = 7.3)
- Cluster 2 (Practical Safety-Conscious Consumers): Lowest liking (Mean = 3.8)

[**SUGGESTED VISUALIZATION: Boxplot showing NanoVan liking by cluster**]

This confirms that Urban Luxury Seekers show the strongest interest in the NanoVan concept, while Safety-Conscious Consumers are the least receptive.

### b. Demographic Profiles

The demographic profiles of each cluster revealed:

**Cluster 0: Value-Conscious Families**
- Age: Younger (Mean = 36.2)
- Income: Lower (Mean = $62,500)
- Miles driven: Moderate (Mean = 12,400)
- Number of kids: Highest (Mean = 2.3)
- Female: Higher percentage (65%)
- Education: Moderate (Mean = 3.1, some college)
- Recycling: High (Mean = 4.2)

**Cluster 1: Urban Luxury Seekers**
- Age: Middle (Mean = 42.1)
- Income: Highest (Mean = $95,300)
- Miles driven: Lowest (Mean = 9,800)
- Number of kids: Low (Mean = 0.8)
- Female: Moderate percentage (48%)
- Education: Highest (Mean = 4.6, college degree+)
- Recycling: Lowest (Mean = 2.8)

**Cluster 2: Practical Safety-Conscious Consumers**
- Age: Oldest (Mean = 48.7)
- Income: Moderate (Mean = $78,600)
- Miles driven: Highest (Mean = 15,700)
- Number of kids: Moderate (Mean = 1.4)
- Female: Lowest percentage (38%)
- Education: Moderate (Mean = 3.5)
- Recycling: High (Mean = 4.5)

[**SUGGESTED VISUALIZATION: Radar chart showing demographic profiles by cluster**]

These demographic profiles align logically with the factor-based needs of each cluster. Value-Conscious Families have more children and lower incomes, explaining their emphasis on practical transportation without luxury. Urban Luxury Seekers have the highest incomes, higher education, and fewer children, consistent with their preference for premium features and compact sizing. Practical Safety-Conscious Consumers are older, drive more miles, and have a higher proportion of males, aligning with their focus on safety and performance features.

## 7. Comparison of Factor Analysis with Principal Component Analysis

We compared our factor analysis results with PCA to understand the differences between these dimension reduction techniques in a marketing context.

[**SUGGESTED VISUALIZATION: Bar chart comparing variance explained by PCA vs. Factor Analysis**]

### Key Differences in Results

1. **Component Loading Patterns**:
   - PCA loadings tended to be more dispersed across variables
   - Factor analysis (with rotation) produced cleaner, more interpretable patterns

2. **Marketing Interpretability**:
   - PCA components were optimal for variance explanation but combined conceptually distinct dimensions
   - Factor analysis better separated conceptually distinct marketing constructs

3. **Strategic Utility**:
   - PCA showed excellent dimensional reduction but was less useful for developing targeted marketing strategies
   - Factor analysis produced dimensions that more directly informed messaging and positioning strategies

4. **Prediction Power**:
   - Both approaches had similar predictive power for NanoVan liking:
     - Original variables: R² = 0.373
     - Factor Analysis: R² = 0.337
     - PCA: R² = 0.329
   - Factor analysis produced more interpretable dimensions that can directly inform marketing strategy

[**SUGGESTED VISUALIZATION: Comparison of regression coefficients between Factor Analysis and PCA**]

## 8. Strategic Recommendations

Based on our comprehensive analysis, we recommend Lake View Associates focus their NanoVan marketing strategy primarily on Cluster 1 (Urban Luxury Seekers) for the following reasons:

1. This segment shows the highest interest in the NanoVan concept (Mean liking = 7.3)
2. The two factors most strongly associated with NanoVan liking (Luxury Orientation and Compact Versatility) perfectly align with this segment's preferences
3. This segment represents higher-income consumers (Mean = $95,300) who are likely willing to pay premium prices, supporting better profit margins
4. Their urban lifestyle and need for compact yet premium vehicles suggests a significant market opportunity that few competitors are addressing

[**SUGGESTED VISUALIZATION: Combined target segment visualization showing demographics, factor profile, and NanoVan liking**]

### Strategic Implementation:

**Product Development:** Emphasize premium materials, sophisticated styling, and high-end technology features while maintaining compact dimensions suitable for urban environments

**Pricing Strategy:** Position as a premium compact offering with pricing that reflects quality and exclusivity

**Marketing Communications:** Focus on messages highlighting:
- Sophisticated styling that stands apart from traditional boxy minivans
- Premium features typically found in luxury sedans
- Smart space utilization that combines compact exterior with versatile interior
- Urban lifestyle compatibility (parking ease, maneuverability)

**Distribution:** Prioritize urban dealerships and showrooms in upscale areas where target consumers shop and live

**Secondary Target:** Consider Cluster 0 (Value-Conscious Families) as a secondary target by offering a more affordable trim level that maintains the compact versatility while reducing luxury features

**Addressing Negative Perceptions:** The significant negative relationship with Factor 5 (Safety and Performance) suggests Lake View Associates should enhance safety messaging and features to address potential concerns in this area

The NanoVan concept represents an opportunity to create a new category of premium compact family vehicles that's currently underserved in the market. By targeting Urban Luxury Seekers who have both the inclination and financial means to purchase such a vehicle, Lake View Associates can maximize the concept's commercial potential.

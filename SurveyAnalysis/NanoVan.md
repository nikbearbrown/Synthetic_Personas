# Factor Analysis in Marketing: The NanoVan Case Study
A comprehensive tutorial on factor analysis and PCA for marketing research

## Introduction to Factor Analysis


Factor analysis is a statistical method used to identify underlying dimensions (factors) that explain correlations among measured variables. In marketing research, it helps uncover latent constructs driving customer preferences.

The mathematical model for Factor Analysis can be represented as:

$X_i = \lambda_{i1}F_1 + \lambda_{i2}F_2 + ... + \lambda_{im}F_m + \varepsilon_i$

Where:
- $X_i$ is the i-th observed variable (survey question)
- $F_1, F_2, ...$ are the common factors
- $\lambda_{ij}$ is the factor loading
- $\varepsilon_i$ is the unique factor (error term)

This notebook demonstrates how to apply factor analysis to the NanoVan case study data.

## Setup and Package Installation
First, we need to install necessary packages and import libraries

```python
# Install required packages (each on its own line to avoid errors)
!pip install factor-analyzer
!pip install scikit-learn
!pip install matplotlib
!pip install seaborn
!pip install pandas
!pip install numpy
!pip install scipy
```

```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set styling for plots
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 8)
```

## Data Loading and Initial Exploration
Let's load the NanoVan survey data and examine its structure

```python
# Load the NanoVan survey data
url = "https://raw.githubusercontent.com/nikbearbrown/Synthetic_Personas/refs/heads/main/SurveyAnalysis/nanovan_data.csv"
df = pd.read_csv(url)
print(f"Dataset shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")

# Display first few rows and basic statistics
print("\nFirst 5 rows:")
print(df.head())
```

## 1. Data Verification
Before proceeding with factor analysis, we need to check data quality, looking for missing values, outliers, and examining distributions.

```python
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Display descriptive statistics for the attribute variables
attribute_vars = [col for col in df.columns if col.startswith('v')]
print("\nDescriptive statistics for attribute variables:")
print(df[attribute_vars].describe())
```

## Visualize Distribution of Attribute Variables
We'll create histograms to examine the distribution of our attribute variables.
This helps identify potential outliers or unusual patterns.

```python
# Create histograms for all attribute variables (v01-v30)
plt.figure(figsize=(15, 10))
df[attribute_vars].hist(figsize=(20, 15), bins=9, color='skyblue', edgecolor='black')
plt.tight_layout()
plt.suptitle('Histograms of Attribute Variables', y=1.02, fontsize=16)
plt.show()
```

## Check for Outliers
Boxplots help visualize potential outliers in our data.

```python
# Create boxplots to check for outliers
plt.figure(figsize=(20, 10))
sns.boxplot(data=df[attribute_vars])
plt.xticks(rotation=90)
plt.title('Boxplots of Attribute Variables')
plt.tight_layout()
plt.show()
```

## Examine Correlation Structure
The correlation matrix helps us understand relationships between variables
and is crucial for determining if factor analysis is appropriate.

```python
# Create correlation matrix heatmap
corr_matrix = df[attribute_vars].corr()
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, 
            square=True, linewidths=.5, cbar_kws={'shrink': .5})
plt.title('Correlation Matrix of Attribute Variables', fontsize=16)
plt.tight_layout()
plt.show()
```

## 2. Establish Relationship with Concept Liking
We'll run a regression analysis to understand how the attribute variables
relate to the overall concept liking (nvliking).

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

X = df[attribute_vars]
y = df['nvliking']

model = LinearRegression()
model.fit(X, y)

# Calculate R-squared and coefficients
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"\nRegression Results:")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
```

## Visualize Regression Coefficients
This shows which attributes have the strongest impact on concept liking.

```python
# Create a dataframe of coefficients
coef_df = pd.DataFrame({
    'Variable': attribute_vars,
    'Coefficient': model.coef_
})
coef_df = coef_df.sort_values('Coefficient', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='Coefficient', y='Variable', data=coef_df)
plt.title('Regression Coefficients: Impact on NanoVan Liking', fontsize=16)
plt.axvline(x=0, color='black', linestyle='--')
plt.grid(axis='x')
plt.tight_layout()
plt.show()
```

## 3. Test Assumptions for Factor Analysis
Before applying factor analysis, we need to verify that our data meets
the necessary assumptions.

```python
# 3A: Determine if factor analysis is appropriate using Bartlett's test and KMO
# Bartlett's test of sphericity
chi2, p = calculate_bartlett_sphericity(df[attribute_vars])
print("\nBartlett's test of sphericity:")
print(f"Chi-square: {chi2:.2f}")
print(f"p-value: {p:.10f}")

# Kaiser-Meyer-Olkin (KMO) Test
kmo_all, kmo_model = calculate_kmo(df[attribute_vars])
print("\nKaiser-Meyer-Olkin (KMO) Test:")
print(f"KMO Score: {kmo_model:.4f}")

if kmo_model < 0.5:
    print("KMO < 0.5: Unacceptable for factor analysis")
elif kmo_model < 0.6:
    print("KMO between 0.5 and 0.6: Miserable")
elif kmo_model < 0.7:
    print("KMO between 0.6 and 0.7: Mediocre")
elif kmo_model < 0.8:
    print("KMO between 0.7 and 0.8: Middling")
elif kmo_model < 0.9:
    print("KMO between 0.8 and 0.9: Meritorious")
else:
    print("KMO ≥ 0.9: Marvelous")
```

## 3B: Determine the Optimal Number of Factors
Multiple methods can help determine how many factors to extract.

```python
# Run initial factor analysis to get eigenvalues
fa = FactorAnalyzer(rotation=None)
fa.fit(df[attribute_vars])
eigenvalues, _ = fa.get_eigenvalues()

# Kaiser criterion (eigenvalue > 1)
num_factors_kaiser = sum(eigenvalues > 1)
print(f"\nNumber of factors with eigenvalue > 1: {num_factors_kaiser}")
```

## Scree Plot for Factor Selection
The scree plot helps visualize eigenvalues and identify the "elbow point"
where additional factors contribute less to explaining variance.

```python
# Create scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
plt.axhline(y=1, color='r', linestyle='--')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot of Eigenvalues')
plt.grid(True)
plt.show()
```

## Variance Explained by Each Factor
This helps us understand how much information is captured by each factor.

```python
# Calculate variance explained by each factor
explained_var = eigenvalues / sum(eigenvalues) * 100
cumulative_var = np.cumsum(explained_var)

# Create a table of variance explained
var_df = pd.DataFrame({
    'Factor': range(1, len(eigenvalues) + 1),
    'Eigenvalue': eigenvalues,
    'Variance Explained (%)': explained_var,
    'Cumulative Variance (%)': cumulative_var
})

print("\nVariance Explained by Factors:")
print(var_df.head(10))  # Show only first 10 factors
```

## 3C: Extract Factors with Varimax Rotation
We'll use Varimax rotation to make the factors more interpretable.

```python
# Extract factors using Varimax rotation
n_factors = num_factors_kaiser  # Using Kaiser criterion
fa_varimax = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
fa_varimax.fit(df[attribute_vars])

# Get loadings
loadings = pd.DataFrame(fa_varimax.loadings_, index=attribute_vars)
loadings.columns = [f'Factor {i+1}' for i in range(n_factors)]

# Highlight significant loadings (> 0.4 in absolute value)
loadings_filtered = loadings.copy()
loadings_filtered[loadings_filtered.abs() < 0.4] = ''

print("\nFactor Loadings (values < 0.4 suppressed for clarity):")
print(loadings_filtered)
```

## Visualize Factor Loadings
A heatmap helps visualize how variables load onto different factors.

```python
# Create heatmap of factor loadings
plt.figure(figsize=(12, 10))
mask = loadings.abs() < 0.4  # Mask insignificant loadings
sns.heatmap(loadings, annot=True, cmap='coolwarm', mask=mask, fmt='.2f', linewidths=.5)
plt.title('Factor Loadings Heatmap (loadings < 0.4 hidden)', fontsize=16)
plt.tight_layout()
plt.show()
```

## 3D: Interpret and Name the Factors
We'll examine which variables load highest on each factor to interpret their meaning.

```python
# Variable descriptions for reference
var_descriptions = {
    'v01': 'kidtrans - Need car to transport kids',
    'v02': 'miniboxy - Current minivans too boxy',
    'v03': 'lthrbetr - Leather seats better than cloth',
    'v04': 'secbiggr - Second car needs to be bigger',
    'v05': 'safeimpt - Auto safety important',
    'v06': 'buyhghnd - Buy higher-end cars',
    'v07': 'pricqual - Price reflects quality',
    'v08': 'prmsound - Premium sound important',
    'v09': 'perfimpt - Performance important',
    'v10': 'tkvacatn - Take many vacations',
    'v11': 'noparkrm - No parking room',
    'v12': 'homlrgst - Home among largest',
    'v13': 'envrminr - Environmental impact minor',
    'v14': 'needbetw - Need between sedan & minivan',
    'v15': 'suvcmpct - SUVs more compact than minivans',
    'v16': 'next2str - Next car will be two-seater',
    'v17': 'carefmny - Careful with money',
    'v18': 'shdcarpl - Should carpool',
    'v19': 'imprtapp - Most appliances imported',
    'v20': 'lk4whldr - Four-wheel drive attractive',
    'v21': 'kidsbulk - Kids take bulky items',
    'v22': 'wntguzlr - Will buy gas guzzler',
    'v23': 'nordtrps - No road trips',
    'v24': 'stylclth - Purchase stylish clothes',
    'v25': 'strngwrn - Strong warranty needed',
    'v26': 'passnimp - Passion more important than pay',
    'v27': 'twoincom - Hard to live on one income',
    'v28': 'nohummer - Not interested in Hummer',
    'v29': 'aftrschl - More after-school activities',
    'v30': 'accesfun - Accessories make car fun'
}

# Examine top variables for each factor
for i in range(n_factors):
    print(f"\nFactor {i+1} - Top Loading Variables:")
    top_vars = loadings.iloc[:, i].abs().sort_values(ascending=False).head(5)
    print(top_vars)
    
    # Print descriptions for top variables
    print("\nVariable Descriptions:")
    for var in top_vars.index:
        print(f"{var}: {var_descriptions[var]}")
```

## Calculate Factor Scores
Factor scores represent how strongly each respondent aligns with each factor.

```python
# Calculate factor scores for each respondent
factor_scores = fa_varimax.transform(df[attribute_vars])
factor_scores_df = pd.DataFrame(factor_scores, 
                               columns=[f'Factor_{i+1}' for i in range(n_factors)])

# Add factor scores to the original dataframe
df_with_factors = pd.concat([df, factor_scores_df], axis=1)
```

## 4. Use Factor Scores to Predict Concept Liking
We'll compare how well the factors predict concept liking compared to the original variables.

```python
# Prepare data
X_factors = factor_scores_df
y = df['nvliking']

# Run regression with factor scores
model_factors = LinearRegression()
model_factors.fit(X_factors, y)

# Calculate R-squared
y_pred_factors = model_factors.predict(X_factors)
r2_factors = r2_score(y, y_pred_factors)
rmse_factors = np.sqrt(mean_squared_error(y, y_pred_factors))

print(f"\nRegression Results with Factor Scores:")
print(f"R-squared: {r2_factors:.4f}")
print(f"RMSE: {rmse_factors:.4f}")

# Compare with original regression
print(f"\nComparison:")
print(f"Original R-squared (30 variables): {r2:.4f}")
print(f"Factor R-squared ({n_factors} factors): {r2_factors:.4f}")
print(f"Difference: {r2 - r2_factors:.4f}")
```

## Factor Importance in Predicting Concept Liking
This shows which factors are most important for predicting NanoVan liking.

```python
# Create factor importance dataframe
factor_importance = pd.DataFrame({
    'Factor': [f'Factor_{i+1}' for i in range(n_factors)],
    'Coefficient': model_factors.coef_,
    'Abs_Coefficient': abs(model_factors.coef_)
})
factor_importance = factor_importance.sort_values('Abs_Coefficient', ascending=False)

print("\nFactor Importance in Predicting NanoVan Liking:")
print(factor_importance)

# Visualize factor importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Factor', data=factor_importance)
plt.title('Factor Importance in Predicting NanoVan Liking', fontsize=16)
plt.axvline(x=0, color='r', linestyle='--')
plt.grid(axis='x')
plt.tight_layout()
plt.show()
```

## 5. Market Segmentation Using Cluster Analysis
Now we'll use the factor scores to segment potential customers.

```python
# 5A: Use hierarchical clustering to help determine number of clusters
from scipy.cluster.hierarchy import dendrogram, linkage

# Calculate linkage
linked = linkage(factor_scores, 'ward')

# Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linked, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.axhline(y=15, color='r', linestyle='--')  # Example threshold line
plt.show()
```

## Use Silhouette Analysis to Determine Optimal Number of Clusters
Silhouette analysis helps validate the optimal number of clusters.

```python
# Determine optimal number of clusters using silhouette score
from sklearn.metrics import silhouette_score

silhouette_scores = []
K = range(2, 8)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(factor_scores)
    silhouette_scores.append(silhouette_score(factor_scores, kmeans.labels_))

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'o-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method for Optimal k', fontsize=16)
plt.grid(True)
plt.show()
```

## 5B: Perform K-means Clustering
Based on the dendrogram and silhouette scores, we'll choose the optimal number of clusters.

```python
# Choose optimal k based on silhouette scores
optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
print(f"Optimal number of clusters: {optimal_k}")

# Run K-means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(factor_scores)

# Add cluster labels to the dataframe
df_with_factors['Cluster'] = clusters
```

## Analyze Cluster Profiles
We'll examine the characteristics of each segment based on their factor scores.

```python
# Calculate mean factor scores by cluster
cluster_profiles = df_with_factors.groupby('Cluster')[factor_scores_df.columns].mean()
print("\nCluster Profiles (Mean Factor Scores):")
print(cluster_profiles)
```

## Visualize Clusters in 2D Space
This shows how the clusters separate in the factor space.

```python
# Plot clusters using the first two factors
plt.figure(figsize=(12, 8))
scatter = plt.scatter(factor_scores[:, 0], factor_scores[:, 1], 
                     c=clusters, cmap='viridis', alpha=0.6, s=50)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.title('Customer Segments Based on Factor Scores', fontsize=16)
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()
```

## 6A: Concept Liking by Cluster
We'll examine how concept liking varies across the identified segments.

```python
# Create boxplot of NanoVan liking by cluster
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='nvliking', data=df_with_factors)
plt.title('NanoVan Liking by Cluster', fontsize=16)
plt.grid(axis='y')
plt.show()

# Test for significant differences using ANOVA
from scipy import stats
f_stat, p_val = stats.f_oneway(
    *[df_with_factors[df_with_factors['Cluster'] == i]['nvliking'] for i in range(optimal_k)]
)
print(f"\nANOVA for nvliking differences between clusters:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_val:.10f}")
```

## 6B: Demographic Profiles of Clusters
We'll analyze the demographic characteristics of each segment.

```python
# Define demographic variables
demographic_vars = ['age', 'income', 'miles', 'numkids', 'female', 'educ', 'recycle']

# Calculate mean demographics by cluster
demographic_summary = df_with_factors.groupby('Cluster')[demographic_vars].mean()
print("\nDemographic Profiles by Cluster:")
print(demographic_summary)
```

## Visualize Demographic Profiles with Radar Chart
This provides a visual comparison of demographics across segments.

```python
# Normalize the demographic data for radar chart
from math import pi

demo_normalized = demographic_summary.copy()
for col in demo_normalized.columns:
    demo_normalized[col] = (demo_normalized[col] - demo_normalized[col].min()) / (demo_normalized[col].max() - demo_normalized[col].min())

# Create radar chart
categories = demographic_vars
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Close the loop

plt.figure(figsize=(12, 10))
ax = plt.subplot(111, polar=True)

# Plot each cluster
for i in range(optimal_k):
    values = demo_normalized.iloc[i].tolist()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {i}')
    ax.fill(angles, values, alpha=0.1)

# Set labels and styling
plt.xticks(angles[:-1], categories)
plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=8)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Demographic Radar Chart by Cluster', fontsize=16)
plt.grid(True)
plt.show()
```

## 7. Strategic Recommendation
Based on our analysis, we'll identify the most promising segment(s) to target.

```python
# Find which cluster(s) like the NanoVan most
cluster_liking = df_with_factors.groupby('Cluster')['nvliking'].mean().sort_values(ascending=False)
print("\nNanoVan Liking by Cluster (Descending):")
print(cluster_liking)
```

## Final Visualization for Strategic Recommendation
This combines key insights to support our targeting recommendation.

```python
# Create combined visualization for the top target segment
plt.figure(figsize=(15, 10))

# Demographic chart for target clusters
plt.subplot(2, 2, 1)
target_cluster = cluster_liking.index[0]  # Top cluster
target_demos = demographic_summary.loc[target_cluster]
plt.bar(target_demos.index, target_demos.values, color='skyblue')
plt.title(f'Demographics of Top Target Cluster ({target_cluster})', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y')

# Factor profile of target cluster
plt.subplot(2, 2, 2)
target_factors = cluster_profiles.loc[target_cluster]
plt.bar(target_factors.index, target_factors.values, color='lightgreen')
plt.title(f'Factor Profile of Top Target Cluster ({target_cluster})', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y')

# Overall distribution of nvliking
plt.subplot(2, 1, 2)
sns.histplot(df['nvliking'], bins=9, kde=True)
plt.axvline(x=df['nvliking'].mean(), color='r', linestyle='--', label='Mean')
plt.title('Overall Distribution of NanoVan Liking', fontsize=14)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```

## Recommendation to Lake View Associates
Based on our factor analysis and cluster analysis, we'll provide strategic recommendations.

```python
print("\nRecommendation to Lake View Associates:")
print("The factor analysis revealed key dimensions underlying customer preferences, with Factors X, Y, and Z being most predictive of NanoVan liking.")
print(f"Cluster {target_cluster} shows the highest interest in the NanoVan concept and has the following profile:")
print(f"- Demographics: {', '.join([f'high {k}' if v > demographic_summary[k].mean() else f'low {k}' for k, v in target_demos.items()])}")
print(f"- Key factors: {', '.join([f'high on {k}' if v > 0 else f'low on {k}' for k, v in target_factors.items()])}")
print("\nWe recommend targeting this segment with marketing messages that emphasize...")
# Final recommendations will be determined after running the analysis
```

## Theoretical Background on Factor Analysis

### Assumptions of Factor Analysis:
1. **Linearity**: Variables have linear relationships
2. **Absence of multicollinearity**: Variables should be correlated but not perfectly
3. **Adequate correlations**: Variables should have sufficient correlations
4. **Adequate sample size**: Generally, 10-15 observations per variable
5. **Absence of outliers**: Extreme values can distort results

### Comparing Factor Analysis with PCA:
| Aspect | Factor Analysis | PCA |
|--------|----------------|-----|
| **Focus** | Shared variance | Total variance |
| **Purpose** | Identify latent constructs | Reduce dimensions |
| **Model** | Assumes underlying factors | Mathematical transformation |
| **Uniqueness** | Accounts for error/uniqueness | Doesn't separate unique variance |
| **Rotation** | Often uses rotation | Typically no rotation |

### Factor Interpretation Tips:
- Look for variables with high loadings (> 0.4 or 0.5)
- Find common themes among variables that load on the same factor
- Consider both positive and negative loadings
- Be cautious with cross-loadings (variables loading on multiple factors)
- Use domain knowledge to make meaningful interpretations

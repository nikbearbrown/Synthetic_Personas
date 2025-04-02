# Autosurvey Analysis Framework and SPSS Assignment

*Nik Bear Brown*.   
*Tuesday, April 1, 2025*

## Introduction

This notebook demonstrates an automated and modular approach to survey data analysis, using the **"SPSS_Assignment.csv"** file as a case study. The broader vision is to create a flexible, extensible framework — **Autosurvey Analysis** — that acts like an AutoML tool, but for survey data. The idea is to enable users to upload *any* structured survey dataset and receive automated, statistically sound insights with minimal manual intervention.

In this specific notebook, we aim to accomplish two things:

1. **Complete the SPSS assignment (IA3: SPSS Assignment) for MKTG 6210: Marketing Research**, following the assignment guidelines provided by Professor Rocklage.
2. **Demonstrate the starting point of an "AutoSurvey" framework**, which can eventually evolve to handle a wide range of survey analysis tasks automatically — such as descriptive stats, group comparisons, correlations, significance testing, and interpretation — with just one or two clicks.

This notebook serves as a dual-purpose prototype:
- **Assignment-focused**: Completing the specific questions using statistical methods and interpreting the results clearly.
- **Framework-focused**: Structuring the analysis pipeline in a modular way to allow scalability and adaptability to future survey datasets and research questions.

---

## Assignment Context

We are provided with a dataset (`SPSS_Assignment.csv`, converted from the original `.sav` format) that contains simulated responses from 100 survey participants. The survey includes the following variables:

- `Gender`: Gender of respondent; 1 = Male, 2 = Female  
- `Location`: Location where respondent lives; 1 = Urban, 2 = Suburban, 3 = Rural  
- `BroadBand`: Broadband internet access; 1 = Yes, 0 = No  
- `SupportStarlink`: 7-point scale measuring support for FCC granting Starlink satellite internet access  
- `SupportSubsidies`: 7-point scale measuring support for federal infrastructure subsidies  

The assignment includes four key research questions (with subparts), each requiring:
1. The correct statistical test selection  
2. Presentation of the output  
3. A clear statistical conclusion  
4. A plain-language interpretation of the findings  

---

## Assignment Questions

**Question 1**: Is there a difference in support for Starlink and Subsidies across genders? If so, describe the mean support by gender.  
**Question 2**: Is there a relationship between support for Starlink and for federal subsidies?  
**Question 3**: Is broadband availability different depending on where people live (Urban, Suburban, Rural)?  
**Question 4**:  
- **Part A**: Is there a significant difference in support for Starlink based on where people live?  
- **Part B**: If so, compare support levels between each location group.  

---

## Goals

- Build an automated process to answer these questions using Python (instead of SPSS), emulating the same statistical rigor.
- Begin laying the foundation for an AutoSurvey Analysis tool — one that adapts to different survey structures and questions, much like AutoML adapts to different modeling problems.

## Dataset Description

The example dataset contains the following variables:

- **Gender**: Gender of respondent; 1 = Male, 2 = Female  
- **Location**: Location where respondent lives; 1 = Urban, 2 = Suburban, 3 = Rural  
- **BroadBand**: Whether the respondent has broadband internet access: 1 = Yes, 0 = No  
- **SupportStarlink**: A 7-point rating scale measuring support for the FCC's granting Starlink use of satellites (1 = Not at all, 7 = Very Much)  
- **SupportSubsidies**: A 7-point rating scale measuring support for federal infrastructure subsidies (1 = Not at all, 7 = Very Much)  

---

### Encoding Categorical Variables Properly: A Note on Gender Coding

In the IA3: SPSS Assignment dataset, the `Gender` variable is encoded as:

- `1 = Male`  
- `2 = Female`

At first glance, this seems fine — it's a common way to store categorical data. But if we apply this directly in a linear regression model, it causes **problems**.

---

### Why Encoding Gender as 1 and 2 Is Problematic

Linear regression treats numeric variables as **quantitative**, meaning it assumes the difference between values is meaningful and linear.

If we fit a model like:

```
y_i = β0 + β1 * Gender_i + ε_i
```

Then the model interprets the difference between Male (`Gender = 1`) and Female (`Gender = 2`) as a **1-unit increase**. That might sound harmless, but here's the problem:

- It assumes a continuous, linear relationship.
- It implies Female is somehow "twice" the category of Male.
- It would (wrongly) let the model predict outcomes for Gender = 1.5, which doesn't exist.

This is **not appropriate for categorical variables**, especially binary ones like gender.

---

### Proper Solution: Dummy or One-Hot Encoding

Instead, we should convert `Gender` into a new variable using **dummy coding**:

```python
Gender_Female = 1 if Gender == 2 else 0
```

Now we can fit the model as:

```
y_i = β0 + β1 · Gender_Female + ε_i
```

This results in:

For males (`Gender_Female = 0`):
```
y_i = β0 + ε_i
```

For females (`Gender_Female = 1`):
```
y_i = β0 + β1 + ε_i
```

In this version:

`β1` represents the difference in average outcome between females and males.

No invalid linear assumptions are made.

Interpretation is valid and meaningful.

---

### AutoSurvey Implication

In an **automated survey analysis framework** (like *AutoSurvey*), this kind of recoding should happen automatically. Here's what the system should do:

- Identify variables with a small number of unique values (e.g., `Gender`, `Location`)
- Determine if they are categorical
- Apply dummy or one-hot encoding internally
- Use models that treat these variables as **factors**, not numbers

This prevents misinterpretation and ensures accurate, robust statistical modeling.

For the same reason dummy variables would be created for Location: Location where respondent lives; 1 = Urban, 2 = Suburban, 3 = Rural

## Analysis Workflow

Our analysis will follow these steps:

1. Data Loading and Preparation
2. Data Type Validation and Cleaning
3. Descriptive Statistics
4. Inferential Statistics
5. Visualization of Results
6. Interpretation and Reporting

Each step is designed to be adaptable to different survey datasets while providing meaningful insights for the current example.

Note that for the full _Autosurvey Tool_ this workflow will be expanded, especially with a lot more data validation, imputation and checks. But for now we'll start with the above.

## Setup and Dependencies

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_ind, pearsonr, chi2_contingency, norm
from IPython.display import display, HTML

# Set visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)

# For reproducibility
np.random.seed(42)
```

## 1. Data Loading and Initial Exploration

```python
# Function to load data from various sources
def load_data(source, file_type='csv'):
    """
    Load survey data from various sources

    Parameters:
    -----------
    source : str
        File path or URL to the data source
    file_type : str
        Type of file ('csv', 'excel', 'spss', etc.)

    Returns:
    --------
    pandas.DataFrame
        Loaded dataset
    """
    if file_type.lower() == 'csv':
        return pd.read_csv(source)
    elif file_type.lower() in ['xls', 'xlsx', 'excel']:
        return pd.read_excel(source)
    # Add support for other file types as needed
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# Load the example dataset
file_path = "https://raw.githubusercontent.com/nikbearbrown/Synthetic_Personas/refs/heads/main/SurveyAnalysis/SPSS_Assignment.csv"  # Update with actual path if needed
try:
    df = load_data(file_path)
    print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
except Exception as e:
    print(f"Error loading data: {e}")

    # For demonstration, create sample data if file not found
    print("Creating sample data for demonstration...")

    # Generate 100 simulated responses
    n_samples = 100

    # Create synthetic data
    data = {
        'Gender': np.random.choice([1, 2], size=n_samples),  # 1=Male, 2=Female
        'Location': np.random.choice([1, 2, 3], size=n_samples),  # 1=Urban, 2=Suburban, 3=Rural
        'BroadBand': np.random.choice([0, 1], size=n_samples),  # 0=No, 1=Yes
        'SupportStarlink': np.random.choice(range(1, 8), size=n_samples),  # 1-7 scale
        'SupportSubsidies': np.random.choice(range(1, 8), size=n_samples)  # 1-7 scale
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Make some realistic patterns in the data
    # Rural areas have less broadband
    for i, row in df.iterrows():
        if row['Location'] == 3:  # Rural
            if np.random.random() < 0.7:  # 70% chance
                df.at[i, 'BroadBand'] = 0

        # Those without broadband tend to support Starlink more
        if row['BroadBand'] == 0:
            df.at[i, 'SupportStarlink'] = min(7, df.at[i, 'SupportStarlink'] + np.random.choice([1, 2]))

    print(f"Sample data created with {df.shape[0]} rows and {df.shape[1]} columns.")

# Display the first few rows
print("\nPreview of the dataset:")
display(df.head())

# Basic information about the dataset
print("\nDataset Information:")
df.info()

# Check for missing values
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print("\nMissing Values:")
    display(missing_values[missing_values > 0])
else:
    print("\nNo missing values found in the dataset.")
```

### No Missing Values Found in the Dataset

If the dataset *did* contain missing values, we would need to handle them appropriately before proceeding with any analysis. Here's how we might approach it:

---

### Dealing with Missing Data (If Any)

#### For **Numeric Variables**  
We could apply imputation strategies such as:

- **KNN Imputation**: Uses the values of the *k* nearest neighbors (based on feature similarity) to fill in missing values. It's a good choice when the data has non-linear relationships or clusters.
  
- **MICE (Multiple Imputation by Chained Equations)**: Iteratively models each variable with missing values as a function of the others. It works well for complex interdependencies in the data.

Both methods are available in Python via `sklearn.impute` and `fancyimpute`.

#### For **Categorical Variables**

Imputing categorical data requires methods that respect the discrete nature of the variable. Good approaches include:

- **Most Frequent (Mode) Imputation**: Replace missing values with the most common category in the column. It's simple and often effective when the mode is dominant.

- **Categorical KNN Imputation**: Like numeric KNN, but adapted to use categorical distances (e.g., Hamming distance) — available in packages like `fancyimpute` or `sklearn-pandas`.

- **Model-Based Imputation**: Predict the missing category using a classification model trained on the non-missing values (e.g., decision tree or logistic regression).

- **Multivariate Imputation with Indicator Variables**: Combine MICE-like methods with indicator variables to flag missingness and preserve information.

---

In our case, **no missing values were found**, so we can proceed directly to analysis. But it's good practice to always check and plan for imputation when building reusable pipelines like those in an AutoSurvey framework.

## 2. Data Type Validation and Configuration

```python
# Define possible variable types for automatic detection
VARIABLE_TYPES = {
    'binary': 'Binary categorical with two values (e.g., Yes/No)',
    'nominal': 'Nominal categorical with unordered categories',
    'ordinal': 'Ordinal categorical with ordered categories',
    'interval': 'Interval scale numeric data',
    'ratio': 'Ratio scale numeric data',
    'id': 'Non-informative identifier'
}

# Function to suggest variable types
def suggest_variable_types(df):
    """
    Suggest appropriate variable types based on data characteristics

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze

    Returns:
    --------
    dict
        Dictionary with column names as keys and suggested types as values
    """
    suggestions = {}

    for col in df.columns:
        unique_values = df[col].nunique()

        # Skip columns with too many unique values (likely IDs)
        if unique_values > 0.9 * len(df):
            suggestions[col] = 'id'
            continue

        # Suggest binary for columns with 2 unique values
        if unique_values == 2:
            suggestions[col] = 'binary'
            continue

        # Check if column contains only integers
        if pd.api.types.is_integer_dtype(df[col]):
            if unique_values <= 7:  # Small number of values
                # Check if values are sequential
                values = sorted(df[col].unique())
                if all(values[i+1] - values[i] == 1 for i in range(len(values)-1)):
                    suggestions[col] = 'ordinal'
                else:
                    suggestions[col] = 'nominal'
            else:
                suggestions[col] = 'interval'
        else:
            # For non-integer numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                suggestions[col] = 'ratio'
            else:
                # For text columns
                suggestions[col] = 'nominal'

    return suggestions

# Define survey-specific variable metadata
def define_variable_metadata(df, suggested_types=None):
    """
    Define metadata for survey variables

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset
    suggested_types : dict, optional
        Dictionary with suggested variable types

    Returns:
    --------
    dict
        Dictionary with comprehensive variable metadata
    """
    if suggested_types is None:
        suggested_types = suggest_variable_types(df)

    # For our example dataset, we know the variable types
    # In a real application, this would be confirmed by the user
    variable_metadata = {
        'Gender': {
            'type': 'nominal',
            'values': {1: 'Male', 2: 'Female'},
            'description': 'Gender of respondent'
        },
        'Location': {
            'type': 'nominal',
            'values': {1: 'Urban', 2: 'Suburban', 3: 'Rural'},
            'description': 'Location where respondent lives'
        },
        'BroadBand': {
            'type': 'binary',
            'values': {0: 'No', 1: 'Yes'},
            'description': 'Whether the respondent has broadband internet access'
        },
        'SupportStarlink': {
            'type': 'ordinal',
            'values': {i: str(i) for i in range(1, 8)},
            'labels': {1: 'Not at all', 7: 'Very Much'},
            'description': 'Support for the FCC granting Starlink use of satellites'
        },
        'SupportSubsidies': {
            'type': 'ordinal',
            'values': {i: str(i) for i in range(1, 8)},
            'labels': {1: 'Not at all', 7: 'Very Much'},
            'description': 'Support for federal subsidies for infrastructure spending'
        }
    }

    return variable_metadata

# Suggest variable types
suggested_types = suggest_variable_types(df)
print("\nSuggested Variable Types:")
for col, var_type in suggested_types.items():
    print(f"- {col}: {var_type}")

# Define metadata
variable_metadata = define_variable_metadata(df, suggested_types)

# Display variable metadata
print("\nVariable Metadata:")
for var, metadata in variable_metadata.items():
    print(f"\n{var}:")
    for key, value in metadata.items():
        if key == 'values' and len(value) > 10:
            print(f"  {key}: {list(value.items())[:3]}... (truncated)")
        else:
            print(f"  {key}: {value}")

# Create labeled versions of categorical variables
df_labeled = df.copy()
for col, metadata in variable_metadata.items():
    if metadata['type'] in ['nominal', 'binary', 'ordinal'] and 'values' in metadata:
        df_labeled[f"{col}_label"] = df[col].map(metadata['values'])

print("\nDataset with labels:")
display(df_labeled.head())
```

## Treating Likert Data: Ordinal or Continuous?

**Likert-scale variables** — such as `SupportStarlink` and `SupportSubsidies` — are a staple in survey research. These use ordered scales (e.g., 1 = "Not at all" to 7 = "Very Much") to capture opinions and attitudes.

A key modeling decision is **how to treat these variables**:

---

### Option 1: Treat as **Ordinal**

Likert items are technically ordinal: they reflect **ranked preferences**, but the **distances between adjacent categories are not guaranteed to be equal**.

#### Pros:
- Statistically conservative and principled.
- Compatible with non-parametric methods (e.g., Mann–Whitney U, Kruskal–Wallis).
- Reduces risk of over-interpreting numeric differences.

#### Cons:
- Limits the use of linear models, means, or Pearson correlations.
- Less powerful for detecting effects when assumptions *could* be met.
- Can complicate modeling in multivariate settings.

---

### Option 2: Treat as **Continuous**

When Likert scales have 5+ points and responses are approximately symmetric or bell-shaped, many practitioners treat them as numeric.

#### Pros:
- Allows use of linear regression, t-tests, Pearson correlation, ANOVA.
- Makes interpretation and communication easier.
- Often valid in practice if scale behaves like an interval.

#### Cons:
- Assumes equal spacing between responses (e.g., 2 → 3 equals 5 → 6).
- May bias results if distribution is skewed or ordinal nature is strong.

---

### What Should AutoSurvey Do?

The power of an **AutoSurvey framework** lies in **not having to choose blindly**. It can:

1. **Detect Likert-type variables** based on structure and metadata.
2. **Run analyses both ways** — treating variables as ordinal *and* as continuous.
3. **Visualize distributions**:
   - Bar charts or histograms by group
   - Boxplots, violin plots, or ECDFs to show shape and spread
4. **Statistically test assumptions**:
   - Use normality tests (e.g., Shapiro-Wilk)
   - Run both parametric and non-parametric versions of key analyses
   - Compare results and flag meaningful differences

---

This dual-analysis strategy makes AutoSurvey:
- More **robust**, by reducing assumption-driven bias
- More **transparent**, by showing how choices impact conclusions
- More **trustworthy**, especially for users without statistical training

> Summary: AutoSurvey doesn't just automate — it **educates** and **audits**. It allows Likert-scale assumptions to be made **visible, testable, and explainable**.

## 3. Descriptive Statistics

```python
# Function to generate descriptive statistics based on variable type
def generate_descriptive_stats(df, metadata):
    """
    Generate appropriate descriptive statistics based on variable type

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    metadata : dict
        Dictionary with variable metadata

    Returns:
    --------
    dict
        Dictionary with descriptive statistics for each variable
    """
    results = {}

    for var, var_meta in metadata.items():
        var_type = var_meta['type']

        if var_type == 'id':
            # Skip ID variables
            continue

        elif var_type in ['nominal', 'binary', 'ordinal']:
            # Categorical variables
            counts = df[var].value_counts().sort_index()
            percentages = df[var].value_counts(normalize=True).sort_index() * 100

            # Add labels if available
            if 'values' in var_meta:
                value_labels = var_meta['values']
                labeled_counts = pd.DataFrame({
                    'Value': counts.index,
                    'Label': [value_labels.get(val, val) for val in counts.index],
                    'Count': counts.values,
                    'Percentage': percentages.values
                })

                results[var] = {
                    'summary': labeled_counts,
                    'mode': df[var].mode()[0],
                    'mode_label': value_labels.get(df[var].mode()[0], df[var].mode()[0])
                }
            else:
                results[var] = {
                    'summary': pd.DataFrame({'Count': counts, 'Percentage': percentages}),
                    'mode': df[var].mode()[0]
                }

            # For ordinal variables, include additional statistics
            if var_type == 'ordinal':
                results[var].update({
                    'mean': df[var].mean(),
                    'median': df[var].median(),
                    'std': df[var].std(),
                    'min': df[var].min(),
                    'max': df[var].max()
                })

        elif var_type in ['interval', 'ratio']:
            # Continuous variables
            results[var] = {
                'summary': df[var].describe(),
                'median': df[var].median(),
                'mode': df[var].mode()[0],
                'skewness': df[var].skew(),
                'kurtosis': df[var].kurtosis()
            }

    return results

# Generate descriptive statistics
descriptive_stats = generate_descriptive_stats(df, variable_metadata)

# Display descriptive statistics
print("\n=== Descriptive Statistics ===")
for var, stats in descriptive_stats.items():
    print(f"\n{var} ({variable_metadata[var]['type']}):")
    print(f"Description: {variable_metadata[var]['description']}")

    if 'summary' in stats:
        display(stats['summary'])

    if variable_metadata[var]['type'] == 'ordinal':
        print(f"Mean: {stats['mean']:.2f}")
        print(f"Median: {stats['median']}")
        print(f"Standard Deviation: {stats['std']:.2f}")

    # Generate appropriate visualizations
    plt.figure(figsize=(10, 6))

    if variable_metadata[var]['type'] in ['nominal', 'binary']:
        # Bar chart for categorical variables
        summary = stats['summary']
        plt.bar(summary['Label'] if 'Label' in summary.columns else summary.index,
                summary['Count'] if 'Count' in summary.columns else summary['Count'])
        plt.title(f"Distribution of {var}")
        plt.ylabel("Count")
        plt.xticks(rotation=45)

    elif variable_metadata[var]['type'] == 'ordinal':
        # Bar chart for ordinal variables with appropriate labels
        summary = stats['summary']
        plt.bar(summary['Label'] if 'Label' in summary.columns else summary.index,
                summary['Count'] if 'Count' in summary.columns else summary['Count'])
        plt.title(f"Distribution of {var}")
        plt.ylabel("Count")
        plt.xlabel(variable_metadata[var]['description'])
        plt.xticks(rotation=45)

    elif variable_metadata[var]['type'] in ['interval', 'ratio']:
        # Histogram for continuous variables
        sns.histplot(df[var], kde=True)
        plt.title(f"Distribution of {var}")
        plt.xlabel(variable_metadata[var]['description'])
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
```

## Summary of Variable Distributions

#### **1. Gender** (Nominal)
- **Type**: Categorical (Nominal)
- **Distribution**:  
  - Male: 42%  
  - Female: 58%
- **Interpretation**:  
  Balanced enough to compare groups, but not numeric — should **not** be treated as continuous.  
  **Modeling note**: Use dummy encoding; no distribution shape applies here.

---

#### **2. Location** (Nominal)
- **Type**: Categorical (Nominal)
- **Distribution**:
  - Urban: 52%  
  - Suburban: 22%  
  - Rural: 26%
- **Interpretation**:  
  Uneven group sizes, but not extreme. Important for group comparisons (e.g., ANOVA).  
  **Modeling note**: Use one-hot encoding or set a reference category. Distribution shape is not relevant.

---

#### **3. BroadBand** (Binary)
- **Type**: Binary Categorical
- **Distribution**:
  - Yes: 66%  
  - No: 34%
- **Interpretation**:  
  Moderate imbalance, but usable. Binary predictor suitable for logistic models or group comparison.  
  **Modeling note**: Treated as a 0/1 indicator variable.

---

#### **4. Support for Starlink** (Ordinal)
- **Mean**: 4.38  
- **Median**: 4.0  
- **Std Dev**: 1.07

- **Distribution Summary**:  
  - Peaks at 4 and 5, tapering off on both ends.  
  - **Fairly symmetric**, with slight concentration around the middle.

- **Interpretation**:  
  Although ordinal, this variable **approximates symmetry** and could be reasonably treated as continuous for parametric tests (e.g., t-test, regression).  
  **AutoSurvey note**: Run both ordinal and continuous models; visually, this could be approximated by a **normal distribution**, pending formal tests.

---

#### **5. Support for Subsidies** (Ordinal)
- **Mean**: 4.20  
- **Median**: 4.0  
- **Std Dev**: 1.07

- **Distribution Summary**:
  - Skewed **to the right** (long tail on higher scores).  
  - Concentrated on values 3–5, with fewer at 6 and 7.

- **Interpretation**:  
  This variable is more **skewed** than SupportStarlink. Using it as continuous may introduce bias unless data is transformed (e.g., log or square root).  
  **AutoSurvey note**: Treat as **ordinal** by default; consider **skewed distribution models** such as:
  - Poisson (for counts),
  - Log-normal,
  - Gamma distribution,
  - Or use non-parametric methods.

---

###  Summary of Modeling Strategy

| Variable             | Type     | Distribution Shape     | Recommended Modeling Approach                  |
|----------------------|----------|-------------------------|-------------------------------------------------|
| Gender               | Nominal  | Not applicable          | Dummy-encoded categorical                       |
| Location             | Nominal  | Not applicable          | One-hot or dummy-encoded                        |
| BroadBand            | Binary   | Not applicable          | Binary (0/1)                                     |
| SupportStarlink      | Ordinal  | Fairly symmetric        | Run both ordinal and continuous models          |
| SupportSubsidies     | Ordinal  | Skewed right            | Prefer ordinal; consider non-parametric methods |

---

>  **AutoSurvey Advantage**:  
> A good framework should **visualize** these distributions, run **normality tests**, and explore both **ordinal and continuous assumptions** — showing whether analytical choices matter for conclusions.

## 4. Inferential Statistics for Research Questions

### Question 1: Gender Differences in Policy Support

```python
def analyze_gender_differences(df, metadata):
    """
    Analyze differences in policy support across genders
    """
    print("\n=== QUESTION 1: Gender Differences in Policy Support ===")

    # Extract the relevant variables
    gender_var = 'Gender'
    policy_vars = ['SupportStarlink', 'SupportSubsidies']

    # Descriptive statistics by gender
    print("\nDescriptive Statistics by Gender:")

    for policy in policy_vars:
        print(f"\n{policy} ({metadata[policy]['description']}):")

        # Group statistics
        gender_groups = df.groupby(gender_var)[policy]
        gender_stats = gender_groups.agg(['count', 'mean', 'std', 'median'])

        # Add gender labels
        gender_stats.index = [metadata[gender_var]['values'][g] for g in gender_stats.index]
        display(gender_stats)

        # Independent samples t-test
        gender_values = df[gender_var].unique()
        group1 = df[df[gender_var] == gender_values[0]][policy]
        group2 = df[df[gender_var] == gender_values[1]][policy]

        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)

        print(f"Independent Samples T-Test Results:")
        print(f"t-statistic: {t_stat:.3f}")
        print(f"p-value: {p_value:.3f}")

        # Effect size (Cohen's d)
        mean1, mean2 = group1.mean(), group2.mean()
        std1, std2 = group1.std(), group2.std()

        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        # Cohen's d
        cohens_d = abs(mean1 - mean2) / pooled_std

        print(f"Effect Size (Cohen's d): {cohens_d:.3f}")

        # Significance interpretation
        alpha = 0.05
        if p_value < alpha:
            print(f"Result: Significant difference in {policy} between genders (p < {alpha})")
            print(f"Interpretation: {metadata[gender_var]['values'][gender_values[0]]}s (Mean = {mean1:.2f}) {'support more' if mean1 > mean2 else 'support less'} than {metadata[gender_var]['values'][gender_values[1]]}s (Mean = {mean2:.2f})")
        else:
            print(f"Result: No significant difference in {policy} between genders (p >= {alpha})")

        # Visualization
        plt.figure(figsize=(10, 6))

        # Boxplot
        sns.boxplot(x=gender_var, y=policy, data=df)
        plt.title(f"Distribution of {policy} by Gender")
        plt.xlabel("Gender")
        plt.ylabel(policy)

        # Add gender labels
        plt.xticks(ticks=range(len(gender_values)),
                  labels=[metadata[gender_var]['values'][g] for g in gender_values])

        plt.tight_layout()
        plt.show()

        # Bar chart with error bars
        plt.figure(figsize=(10, 6))

        # Calculate means and confidence intervals
        means = gender_groups.mean()
        cis = [norm.interval(0.95, loc=mean, scale=std/np.sqrt(count))
               for mean, std, count in zip(gender_stats['mean'], gender_stats['std'], gender_stats['count'])]

        errors = [[mean - ci[0] for mean, ci in zip(means, cis)],
                 [ci[1] - mean for mean, ci in zip(means, cis)]]

        # Plot
        x_pos = np.arange(len(gender_values))
        plt.bar(x_pos, means, yerr=errors, align='center', alpha=0.7, capsize=10)
        plt.xticks(x_pos, [metadata[gender_var]['values'][g] for g in gender_values])
        plt.ylabel(f"Mean {policy} Score")
        plt.title(f"Mean {policy} Score by Gender with 95% CI")

        plt.tight_layout()
        plt.show()

    # Conclusion and interpretation
    print("\nConclusion and Interpretation:")
    print("Based on the statistical analysis, we can determine whether gender significantly influences support for Starlink and federal subsidies policies.")
    print("The t-test results show whether there are statistically significant differences between males and females in their support levels.")
    print("The effect size (Cohen's d) indicates the practical significance of any differences found.")

# Run the gender difference analysis
analyze_gender_differences(df, variable_metadata)
```

# Analysis of Gender Differences in Policy Support (Question 1)

## Research Question
Question 1 examined whether there are significant differences between male and female respondents in their support for two policy areas: the FCC granting Starlink use of satellites for internet service and federal subsidies for infrastructure spending.

## Methodology
An independent samples t-test with Welch's correction (equal variances not assumed) was conducted to compare mean support scores between genders for each policy area. The analysis included:
- Descriptive statistics by gender
- Statistical significance testing (α = 0.05)
- Effect size calculation using Cohen's d

## Results for Starlink Support

### Descriptive Statistics
- **Males (n=42)**: Mean = 4.40, Standard Deviation = 1.17, Median = 4.0
- **Females (n=58)**: Mean = 4.36, Standard Deviation = 1.00, Median = 4.0

### Inferential Statistics
- **t-statistic**: -0.191
- **p-value**: 0.849
- **Effect Size (Cohen's d)**: 0.040 (very small)

### Interpretation
No statistically significant difference was detected between males and females in their support for the FCC granting Starlink use of satellites (p > 0.05). The extremely small effect size (d = 0.040) further confirms that any difference between genders is negligible. Both gender groups showed moderate support for Starlink, with nearly identical means and medians. The standard deviation indicates slightly more variability among male respondents than females.

## Results for Federal Infrastructure Subsidies

### Descriptive Statistics
- **Males (n=42)**: Mean = 4.48, Standard Deviation = 0.92, Median = 5.0
- **Females (n=58)**: Mean = 4.00, Standard Deviation = 1.14, Median = 4.0

### Inferential Statistics
- **t-statistic**: -2.312
- **p-value**: 0.023
- **Effect Size (Cohen's d)**: 0.453 (medium)

### Interpretation
A statistically significant difference was found between males and females in their support for federal infrastructure subsidies (p < 0.05). Males reported significantly higher support (M = 4.48) than females (M = 4.00). The effect size (d = 0.453) indicates a medium practical significance to this finding. The difference in median values (5.0 for males vs. 4.0 for females) further supports this conclusion. The standard deviation indicates more variability in female responses compared to male responses.

## Conclusion
Gender appears to be a factor in determining support for federal infrastructure subsidies but not for Starlink satellite internet. Males show significantly higher support for infrastructure subsidies than females, with a medium effect size suggesting this difference is not only statistically significant but also practically meaningful.

The findings suggest that communication and policy strategies regarding infrastructure spending may need to consider gender differences, while approaches regarding satellite internet technologies like Starlink could be more uniform across genders. The higher variability in female responses to the infrastructure subsidies question also suggests more diverse opinions within this demographic group.

These results contribute to our understanding of how demographic factors may influence public support for different types of technology and infrastructure policies.

### Question 2: Relationship Between Support for Starlink and Subsidies

```python
def analyze_policy_relationship(df, metadata):
    """
    Analyze the relationship between support for Starlink and federal subsidies
    """
    print("\n=== QUESTION 2: Relationship Between Support for Starlink and Federal Subsidies ===")

    # Extract the relevant variables
    policy1 = 'SupportStarlink'
    policy2 = 'SupportSubsidies'

    # Descriptive statistics
    print("\nDescriptive Statistics:")
    display(df[[policy1, policy2]].describe())

    # Correlation analysis
    corr, p_value = pearsonr(df[policy1], df[policy2])

    print(f"\nPearson Correlation Results:")
    print(f"Correlation coefficient (r): {corr:.3f}")
    print(f"p-value: {p_value:.3f}")

    # Interpretation of correlation strength
    if abs(corr) < 0.3:
        strength = "weak"
    elif abs(corr) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"

    direction = "positive" if corr > 0 else "negative"

    # Significance interpretation
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: Significant {strength} {direction} correlation (p < {alpha})")
        print(f"Interpretation: There is a {strength} {direction} relationship between support for Starlink and support for federal subsidies.")
        print(f"As support for one policy increases, support for the other {'increases' if direction == 'positive' else 'decreases'}.")
    else:
        print(f"Result: No significant correlation (p >= {alpha})")
        print("Interpretation: There is no significant relationship between support for Starlink and support for federal subsidies.")

    # Visualization
    plt.figure(figsize=(10, 8))

    # Scatter plot with regression line
    sns.regplot(x=policy1, y=policy2, data=df, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})

    plt.title(f"Relationship Between {policy1} and {policy2}")
    plt.xlabel(metadata[policy1]['description'])
    plt.ylabel(metadata[policy2]['description'])

    # Add correlation information to plot
    plt.annotate(f"r = {corr:.2f}, p = {p_value:.3f}",
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.show()

    # 2D Histogram / Heatmap
    plt.figure(figsize=(10, 8))

    # Create a cross-tabulation
    crosstab = pd.crosstab(df[policy1], df[policy2])

    # Heatmap
    sns.heatmap(crosstab, annot=True, cmap="YlGnBu", fmt="d", cbar_kws={'label': 'Count'})
    plt.title(f"Frequency of {policy1} and {policy2} Combinations")
    plt.xlabel(metadata[policy2]['description'])
    plt.ylabel(metadata[policy1]['description'])

    plt.tight_layout()
    plt.show()

    # Conclusion
    print("\nConclusion:")
    if p_value < alpha:
        print(f"There is a statistically significant {strength} {direction} correlation (r = {corr:.2f}, p = {p_value:.3f}) between support for Starlink and federal subsidies.")
        print(f"This suggests that respondents who {'support' if direction == 'positive' else 'oppose'} one policy tend to {'support' if direction == 'positive' else 'oppose'} the other as well.")
    else:
        print("There is no statistically significant correlation between support for Starlink and federal subsidies.")
        print("This suggests that support for these two policies is not related in our sample.")

# Run the policy relationship analysis
analyze_policy_relationship(df, variable_metadata)
```

# Analysis of the Relationship Between Support for Starlink and Federal Subsidies (Question 2)

## Research Question
Question 2 investigated whether there is a significant relationship between respondents' support for the FCC granting Starlink use of satellites and their support for federal infrastructure subsidies.

## Methodology
A Pearson correlation analysis was conducted to examine the linear relationship between these two policy support variables. The analysis included:
- Descriptive statistics for both variables
- Correlation coefficient calculation
- Statistical significance testing (α = 0.05)
- Strength and direction interpretation

## Results

### Descriptive Statistics
| Statistic | Support for Starlink | Support for Federal Subsidies |
|-----------|----------------------|-------------------------------|
| Count     | 100.00               | 100.00                        |
| Mean      | 4.38                 | 4.20                          |
| Std       | 1.07                 | 1.07                          |
| Min       | 2.00                 | 1.00                          |
| 25%       | 4.00                 | 3.00                          |
| 50% (Median) | 4.00              | 4.00                          |
| 75%       | 5.00                 | 5.00                          |
| Max       | 7.00                 | 6.00                          |

Both support variables show similar central tendencies with identical medians (4.00) and similar means (4.38 for Starlink, 4.20 for subsidies). They also display identical standard deviations (1.07), indicating comparable variability in responses. The distributions are slightly positively skewed, with support for Starlink reaching a maximum of 7 (very high support) while support for subsidies peaked at 6.

### Inferential Statistics
- **Correlation coefficient (r)**: 0.276
- **p-value**: 0.005

### Interpretation
The analysis revealed a statistically significant positive correlation between support for Starlink and support for federal subsidies (r = 0.276, p = 0.005). This correlation is considered weak in magnitude (r < 0.3), but is statistically significant at the p < 0.01 level.

The positive direction indicates that as support for one policy increases, support for the other tends to increase as well. This suggests some alignment in respondents' attitudes toward these different types of technology and infrastructure initiatives.

## Implications
The weak positive correlation indicates that while there is some relationship between attitudes toward these policies, they are largely distinct constructs with considerable independent variance. The shared variance (r² = 0.076) is only about 7.6%, meaning that 92.4% of the variance in support for one policy cannot be explained by support for the other.

This finding suggests that:

1. There may be a general predisposition among some respondents to support technology and infrastructure initiatives broadly, regardless of the specific type.

2. Despite this general tendency, people largely form their opinions about Starlink and federal subsidies independently, likely based on different factors or priorities.

3. Communication strategies should not assume that support for one type of initiative automatically translates to support for the other.

## Conclusion
The statistically significant weak positive correlation between support for Starlink and federal subsidies provides evidence that these attitudes are somewhat related but largely distinct. While respondents who support one policy tend to show slightly higher support for the other, the weak magnitude of this relationship suggests that different factors likely influence each type of support. This understanding could help policymakers and stakeholders develop more targeted and effective communication strategies for each initiative.

### Question 3: Broadband Availability Across Regions

```python
def analyze_broadband_by_location(df, metadata):
    """
    Analyze broadband availability across different regions
    """
    print("\n=== QUESTION 3: Broadband Availability Across Regions ===")

    # Extract the relevant variables
    location_var = 'Location'
    broadband_var = 'BroadBand'

    # Create a cross-tabulation
    crosstab = pd.crosstab(df[location_var], df[broadband_var])

    # Add percentage
    crosstab_pct = pd.crosstab(df[location_var], df[broadband_var], normalize='index') * 100

    # Add labels
    crosstab.columns = [metadata[broadband_var]['values'][col] for col in crosstab.columns]
    crosstab.index = [metadata[location_var]['values'][idx] for idx in crosstab.index]

    crosstab_pct.columns = [metadata[broadband_var]['values'][col] for col in crosstab_pct.columns]
    crosstab_pct.index = [metadata[location_var]['values'][idx] for idx in crosstab_pct.index]

    print("\nBroadband Availability by Location (Counts):")
    display(crosstab)

    print("\nBroadband Availability by Location (Percentages):")
    display(crosstab_pct)

    # Chi-square test of independence
    chi2, p_value, dof, expected = chi2_contingency(crosstab)

    print(f"\nChi-square Test Results:")
    print(f"Chi-square statistic: {chi2:.3f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p_value:.3f}")

    # Effect size (Cramer's V)
    n = crosstab.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))

    print(f"Effect Size (Cramer's V): {cramers_v:.3f}")

    # Significance interpretation
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: Significant association between location and broadband availability (p < {alpha})")

        # Post-hoc analysis: standardized residuals
        observed = crosstab.values

        # Calculate standardized residuals
        standardized_residuals = (observed - expected) / np.sqrt(expected)

        # Create DataFrame for residuals
        residuals_df = pd.DataFrame(standardized_residuals,
                                   index=crosstab.index,
                                   columns=crosstab.columns)

        print("\nStandardized Residuals:")
        display(residuals_df)

        # Interpretation of significant cells
        print("\nSignificant Associations:")
        for i, location in enumerate(crosstab.index):
            for j, broadband in enumerate(crosstab.columns):
                if abs(standardized_residuals[i, j]) > 1.96:  # 95% confidence
                    direction = "more" if standardized_residuals[i, j] > 0 else "less"
                    print(f"- {location} has significantly {direction} {broadband} broadband than expected (z = {standardized_residuals[i, j]:.2f})")
    else:
        print(f"Result: No significant association between location and broadband availability (p >= {alpha})")

    # Visualization
    plt.figure(figsize=(12, 8))

    # Grouped bar chart
    crosstab.plot(kind='bar', figsize=(12, 8))
    plt.title("Broadband Availability by Location")
    plt.xlabel("Location")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.legend(title="Broadband Access")

    plt.tight_layout()
    plt.show()

    # Stacked percentage bar chart
    plt.figure(figsize=(12, 8))

    crosstab_pct.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title("Percentage of Broadband Availability by Location")
    plt.xlabel("Location")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.legend(title="Broadband Access")

    # Add percentage labels
    for i, location in enumerate(crosstab_pct.index):
        for j, col in enumerate(crosstab_pct.columns):
            value = crosstab_pct.loc[location, col]
            if value > 5:  # Only show labels for segments > 5%
                plt.annotate(f'{value:.1f}%',
                           xy=(i, crosstab_pct.loc[location].cumsum()[j] - value/2),
                           ha='center', va='center',
                           fontweight='bold', color='white')

    plt.tight_layout()
    plt.show()

    # Conclusion
    print("\nConclusion:")
    if p_value < alpha:
        print(f"There is a statistically significant association between location and broadband availability (χ² = {chi2:.2f}, p = {p_value:.3f}, Cramer's V = {cramers_v:.3f}).")

        # Describe the pattern based on percentages
        highest_yes = crosstab_pct['Yes'].idxmax()
        lowest_yes = crosstab_pct['Yes'].idxmin()

        print(f"The data shows that broadband access varies significantly by location, with {highest_yes} areas having the highest broadband availability ({crosstab_pct.loc[highest_yes, 'Yes']:.1f}%) and {lowest_yes} areas having the lowest ({crosstab_pct.loc[lowest_yes, 'Yes']:.1f}%).")
    else:
        print(f"There is no statistically significant association between location and broadband availability (χ² = {chi2:.2f}, p = {p_value:.3f}).")
        print("This suggests that in our sample, broadband access is similarly distributed across urban, suburban, and rural areas.")

# Run the broadband by location analysis
analyze_broadband_by_location(df, variable_metadata)
```

# Detailed Analysis of Broadband Availability Across Regions (Question 3)

## Research Question
Question 3 investigated whether broadband internet access varies significantly across different geographic locations (urban, suburban, and rural areas), testing the hypothesis that rural areas have less access to broadband internet.

## Methodology
The analysis employed a chi-square test of independence to examine the relationship between location and broadband availability. The methodology included:
- Cross-tabulation of broadband access by location (counts and percentages)
- Chi-square test of independence
- Effect size calculation using Cramer's V
- Post-hoc analysis using standardized residuals
- Visual representation through bar charts and stacked percentage charts

## Results

### Descriptive Statistics
The cross-tabulation revealed substantial differences in broadband availability across locations:

Broadband Availability by Location (Counts):

Urban:
- No broadband: 3 households
- Yes broadband: 49 households
- Total: 52 households

Suburban:
- No broadband: 11 households
- Yes broadband: 11 households
- Total: 22 households

Rural:
- No broadband: 20 households
- Yes broadband: 6 households
- Total: 26 households


Broadband Availability by Location (Percentages):

Urban:
- No broadband: 5.77%
- Yes broadband: 94.23%

Suburban:
- No broadband: 50.00%
- Yes broadband: 50.00%

Rural:
- No broadband: 76.92%
- Yes broadband: 23.08%

The percentage analysis shows a striking gradient in broadband access:
- Urban areas enjoy near-universal broadband access (94.23%)
- Suburban areas have balanced access (50% with broadband)
- Rural areas have severely limited access (only 23.08% with broadband)

### Inferential Statistics
- **Chi-square statistic**: 42.325
- **Degrees of freedom**: 2
- **p-value**: 0.000 (highly significant)
- **Effect Size (Cramer's V)**: 0.651 (strong effect)

The chi-square test revealed a highly significant association between location and broadband availability (p < 0.001). The Cramer's V value of 0.651 indicates a strong association between these variables, suggesting that location is a powerful predictor of broadband access.

### Post-hoc Analysis
Standardized residuals identified specific location-broadband combinations that significantly contributed to the overall chi-square result:

```
Standardized Residuals:

Urban:
- No broadband: -3.49 (significantly less than expected)
- Yes broadband: 2.51 (significantly more than expected)

Suburban:
- No broadband: 1.29 (not significant)
- Yes broadband: -0.92 (not significant)

Rural:
- No broadband: 3.75 (significantly more than expected)
- Yes broadband: -2.69 (significantly less than expected)
```

Significant associations (where |z| > 1.96) included:
- Urban areas had significantly fewer "No" broadband responses than expected (z = -3.49)
- Urban areas had significantly more "Yes" broadband responses than expected (z = 2.51)
- Rural areas had significantly more "No" broadband responses than expected (z = 3.75)
- Rural areas had significantly fewer "Yes" broadband responses than expected (z = -2.69)

Notably, suburban areas did not show statistically significant deviations from expected values, suggesting they represent a middle ground in broadband availability.

## Conclusion
The analysis confirmed a statistically significant and strong association between geographic location and broadband availability (χ² = 42.32, p < 0.001, Cramer's V = 0.651).

Urban areas had the highest broadband availability (94.2%), followed by suburban areas (50.0%), while rural areas had dramatically lower access (23.1%). This creates a clear digital divide across geographic locations, with rural residents at a significant disadvantage in terms of internet connectivity.

These findings strongly support the hypothesis that broadband access varies by location, with rural areas being particularly underserved. The magnitude of the disparity (over 70 percentage points difference between urban and rural access) and the strong effect size underscore the severity of this digital divide.

This clear evidence of broadband inequality has important implications for policy decisions, suggesting that targeted initiatives to expand broadband infrastructure in rural areas should be prioritized to address this significant technological gap. The digital divide identified may substantially impact educational opportunities, economic development, healthcare access, and quality of life for residents in rural communities.

### Question 4: Support for Starlink Across Different Locations

#### Part A: Overall Differences in Support by Location
#### Part B: Pairwise Comparisons Between Locations

```python
def analyze_starlink_by_location(df, metadata):
    """
    Analyze differences in support for Starlink across different locations
    """
    print("\n=== QUESTION 4: Support for Starlink Across Different Locations ===")

    # Extract the relevant variables
    location_var = 'Location'
    starlink_var = 'SupportStarlink'

    # Part A: Overall differences in support by location
    print("\n=== Part A: Overall Differences in Support by Location ===\n")

    # Descriptive statistics by location
    location_groups = df.groupby(location_var)[starlink_var]
    location_stats = location_groups.agg(['count', 'mean', 'std', 'median', 'min', 'max'])

    # Add location labels
    location_stats.index = [metadata[location_var]['values'][loc] for loc in location_stats.index]

    print("Descriptive Statistics of Starlink Support by Location:")
    display(location_stats)

    # Check assumptions for ANOVA
    # 1. Normality within groups (Shapiro-Wilk test)
    print("\nTesting Normality Within Groups:")
    from scipy.stats import shapiro

    normality_results = {}
    for location in df[location_var].unique():
        loc_data = df[df[location_var] == location][starlink_var]
        stat, p_value = shapiro(loc_data)
        normality_results[metadata[location_var]['values'][location]] = {
            'W-statistic': stat,
            'p-value': p_value,
            'Normal': p_value > 0.05
        }

    for location, results in normality_results.items():
        print(f"{location}: W = {results['W-statistic']:.3f}, p = {results['p-value']:.3f}, " +
              f"{'Normally distributed' if results['Normal'] else 'Not normally distributed'}")

    # 2. Homogeneity of variances (Levene's test)
    from scipy.stats import levene

    location_groups_data = [df[df[location_var] == loc][starlink_var] for loc in df[location_var].unique()]
    levene_stat, levene_p = levene(*location_groups_data)

    print(f"\nLevene's Test for Homogeneity of Variances:")
    print(f"Statistic: {levene_stat:.3f}, p-value: {levene_p:.3f}")
    print(f"{'Equal variances' if levene_p > 0.05 else 'Unequal variances'}")

    # One-way ANOVA
    from scipy.stats import f_oneway

    f_stat, p_value = f_oneway(*location_groups_data)

    print("\nOne-way ANOVA Results:")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"p-value: {p_value:.3f}")

    # Alternative: Non-parametric Kruskal-Wallis H-test
    from scipy.stats import kruskal

    try:
        h_stat, h_p_value = kruskal(*location_groups_data)
        print("\nKruskal-Wallis H-test Results (Non-parametric alternative):")
        print(f"H-statistic: {h_stat:.3f}")
        print(f"p-value: {h_p_value:.3f}")

        # Use the non-parametric result if assumptions for ANOVA are violated
        if not all(result['Normal'] for result in normality_results.values()) or levene_p <= 0.05:
            print("\nNote: Using non-parametric test results due to violation of ANOVA assumptions.")
            overall_result = h_p_value
        else:
            overall_result = p_value
    except Exception as e:
        print(f"\nCould not perform Kruskal-Wallis test: {e}")
        overall_result = p_value

    # Overall significance interpretation
    alpha = 0.05
    if overall_result < alpha:
        print(f"\nResult: Significant differences in Starlink support across locations (p < {alpha})")
        print(f"Interpretation: Support for Starlink varies significantly based on where people live.")
    else:
        print(f"\nResult: No significant differences in Starlink support across locations (p >= {alpha})")
        print("Interpretation: Support for Starlink does not vary significantly based on where people live.")

    # Visualization
    plt.figure(figsize=(12, 6))

    # Boxplot
    sns.boxplot(x=location_var, y=starlink_var, data=df)
    plt.title(f"Distribution of Support for Starlink by Location")
    plt.xlabel("Location")
    plt.ylabel("Support for Starlink")

    # Add location labels
    plt.xticks(ticks=range(len(df[location_var].unique())),
              labels=[metadata[location_var]['values'][loc] for loc in sorted(df[location_var].unique())])

    plt.tight_layout()
    plt.show()

    # Bar chart with error bars
    plt.figure(figsize=(12, 6))

    # Calculate means and confidence intervals
    means = location_groups.mean()

    # For confidence intervals
    cis = [norm.interval(0.95, loc=mean, scale=std/np.sqrt(count))
           for mean, std, count in zip(location_stats['mean'], location_stats['std'], location_stats['count'])]

    errors = [[mean - ci[0] for mean, ci in zip(means, cis)],
             [ci[1] - mean for mean, ci in zip(means, cis)]]

    # Plot
    x_pos = np.arange(len(means))
    plt.bar(x_pos, means, yerr=errors, align='center', alpha=0.7, capsize=10)
    plt.xticks(x_pos, [metadata[location_var]['values'][loc] for loc in sorted(means.index)])
    plt.ylabel(f"Mean Support for Starlink Score")
    plt.title(f"Mean Support for Starlink by Location with 95% CI")

    plt.tight_layout()
    plt.show()

    # Part B: Pairwise comparisons
    print("\n=== Part B: Pairwise Comparisons Between Locations ===\n")

    # Only proceed with pairwise comparisons if overall test is significant
    if overall_result < alpha:
        # Post-hoc tests
        # For parametric: Tukey's HSD (Honest Significant Difference)
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        # Prepare data for Tukey's test
        tukey_data = df[starlink_var].values
        tukey_labels = df[location_var].map(metadata[location_var]['values']).values

        # Perform Tukey's test
        tukey_results = pairwise_tukeyhsd(tukey_data, tukey_labels, alpha=0.05)

        print("Tukey's HSD Post-hoc Test Results:")
        print(tukey_results)

        # Alternative: Non-parametric Mann-Whitney U test for each pair
        from scipy.stats import mannwhitneyu

        print("\nMann-Whitney U Test for Pairwise Comparisons:")

        # Get all unique pairs of locations
        locations = sorted(df[location_var].unique())
        location_names = [metadata[location_var]['values'][loc] for loc in locations]

        # For effect size calculation (r = Z / sqrt(N))
        def calculate_effect_size(u_stat, n1, n2):
            # Calculate Z-score from U statistic
            mean_u = n1 * n2 / 2
            std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            z_score = (u_stat - mean_u) / std_u
            # Calculate effect size r
            r = z_score / np.sqrt(n1 + n2)
            return r

        # Bonferroni correction for multiple comparisons
        num_comparisons = len(locations) * (len(locations) - 1) // 2
        adjusted_alpha = alpha / num_comparisons

        for i, loc1 in enumerate(locations):
            for j, loc2 in enumerate(locations):
                if i < j:  # Ensure we only compare each pair once
                    group1 = df[df[location_var] == loc1][starlink_var]
                    group2 = df[df[location_var] == loc2][starlink_var]

                    # Mann-Whitney U test
                    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

                    # Effect size
                    n1, n2 = len(group1), len(group2)
                    effect_size = calculate_effect_size(u_stat, n1, n2)

                    print(f"{location_names[i]} vs {location_names[j]}:")
                    print(f"  U-statistic: {u_stat:.3f}")
                    print(f"  p-value: {p_value:.3f}")
                    print(f"  Effect size (r): {effect_size:.3f}")

                    # Interpretation with Bonferroni correction
                    if p_value < adjusted_alpha:
                        print(f"  Result: Significant difference (p < {adjusted_alpha:.4f}, Bonferroni-corrected)")
                        mean1, mean2 = group1.mean(), group2.mean()
                        print(f"  {location_names[i]} (Mean = {mean1:.2f}) {'has higher' if mean1 > mean2 else 'has lower'} support than {location_names[j]} (Mean = {mean2:.2f})")
                    else:
                        print(f"  Result: No significant difference (p >= {adjusted_alpha:.4f}, Bonferroni-corrected)")
                    print()

        # Conclusion for pairwise comparisons
        print("\nSummary of Pairwise Comparisons:")
        print("Based on the post-hoc analyses, we can identify which specific pairs of locations differ significantly in their support for Starlink.")

        # Identify the highest and lowest support groups
        highest_support = location_stats['mean'].idxmax()
        lowest_support = location_stats['mean'].idxmin()

        print(f"\nHighest support: {highest_support} (Mean = {location_stats.loc[highest_support, 'mean']:.2f})")
        print(f"Lowest support: {lowest_support} (Mean = {location_stats.loc[lowest_support, 'mean']:.2f})")
    else:
        print("Since there are no significant overall differences in Starlink support across locations,")
        print("pairwise comparisons are not necessary or recommended.")

    # Overall conclusion
    print("\nConclusion for Question 4:")
    if overall_result < alpha:
        print(f"There are statistically significant differences in support for Starlink based on location.")
        print(f"The post-hoc analyses help us understand which specific location pairs differ from each other.")

        # If we have specific findings from the pairwise tests, add them here
        # For example: "Rural residents show significantly higher support for Starlink compared to Urban residents..."
    else:
        print(f"There are no statistically significant differences in support for Starlink based on location.")
        print(f"This suggests that geographic location is not a major factor in determining support for Starlink in our sample.")

# Run the analysis
analyze_starlink_by_location(df, variable_metadata)
```

# Detailed Analysis of Support for Starlink Across Different Locations (Question 4)

## Research Question
Question 4 examined whether support for the FCC granting Starlink the right to provide satellite-based internet varies significantly based on where people live (urban, suburban, or rural areas), and if differences exist, which specific location pairs differ from each other.

## Methodology
The analysis was conducted in two parts:
1. Part A examined overall differences using ANOVA and non-parametric alternatives
2. Part B performed pairwise comparisons between location groups

Statistical approaches included:
- Assumption testing (normality and homogeneity of variance)
- One-way ANOVA
- Kruskal-Wallis H-test (non-parametric alternative)
- Post-hoc analyses (Tukey's HSD and Mann-Whitney U tests with Bonferroni correction)
- Effect size calculations

## Results: Part A - Overall Differences

### Descriptive Statistics
| Location  | Count | Mean | Std  | Median | Min | Max |
|-----------|-------|------|------|--------|-----|-----|
| Urban     | 52    | 4.17 | 1.04 | 4.0    | 2   | 6   |
| Suburban  | 22    | 4.14 | 0.89 | 4.0    | 2   | 6   |
| Rural     | 26    | 5.00 | 1.06 | 5.0    | 3   | 7   |

Rural residents showed notably higher support for Starlink (mean = 5.00) compared to urban (mean = 4.17) and suburban (mean = 4.14) residents. Rural areas also had a higher minimum value (3 vs. 2) and maximum value (7 vs. 6), indicating a generally more positive attitude toward Starlink.

### Assumption Testing
1. **Normality Testing (Shapiro-Wilk):**
   - Urban: W = 0.893, p = 0.000 (violation of normality)
   - Rural: W = 0.900, p = 0.015 (violation of normality)
   - Suburban: W = 0.890, p = 0.019 (violation of normality)

2. **Homogeneity of Variances (Levene's test):**
   - Statistic = 1.635, p-value = 0.200 (equal variances assumption met)

Due to the violation of the normality assumption, the non-parametric Kruskal-Wallis H-test was deemed more appropriate than ANOVA.

### Inferential Statistics
1. **One-way ANOVA:**
   - F-statistic = 6.558
   - p-value = 0.002 (significant)

2. **Kruskal-Wallis H-test:**
   - H-statistic = 10.590
   - p-value = 0.005 (significant)

Both parametric and non-parametric tests confirmed significant differences in Starlink support across locations (p < 0.01), indicating that geographic location significantly influences attitudes toward Starlink satellite internet.

## Results: Part B - Pairwise Comparisons

### Tukey's HSD Post-hoc Test
| Group 1  | Group 2   | Mean Diff | p-adj  | Lower   | Upper   | Reject |
|----------|-----------|-----------|--------|---------|---------|--------|
| Rural    | Suburban  | -0.8636   | 0.0115 | -1.5638 | -0.1634 | True   |
| Rural    | Urban     | -0.8269   | 0.0029 | -1.4075 | -0.2463 | True   |
| Suburban | Urban     | 0.0367    | 0.9889 | -0.578  | 0.6515  | False  |

The Tukey's HSD test revealed:
1. Rural residents showed significantly higher support than suburban residents (p = 0.0115)
2. Rural residents showed significantly higher support than urban residents (p = 0.0029)
3. No significant difference between suburban and urban residents (p = 0.9889)

### Mann-Whitney U Tests (with Bonferroni correction)
1. **Urban vs. Suburban:**
   - U-statistic = 587.500
   - p-value = 0.853
   - Effect size (r) = 0.021
   - Result: No significant difference (p ≥ 0.0167)

2. **Urban vs. Rural:**
   - U-statistic = 405.000
   - p-value = 0.003
   - Effect size (r) = -0.325 (medium effect)
   - Result: Significant difference (p < 0.0167)
   - Urban (Mean = 4.17) had lower support than Rural (Mean = 5.00)

3. **Suburban vs. Rural:**
   - U-statistic = 159.000
   - p-value = 0.006
   - Effect size (r) = -0.379 (medium effect)
   - Result: Significant difference (p < 0.0167)
   - Suburban (Mean = 4.14) had lower support than Rural (Mean = 5.00)

The non-parametric pairwise comparisons corroborated the Tukey's HSD findings, confirming that rural residents have significantly higher support for Starlink compared to both urban and suburban residents, with medium effect sizes.

## Conclusion
The analysis provided strong evidence that support for Starlink varies significantly based on location (p < 0.01). Rural residents demonstrated substantially higher support (Mean = 5.00) compared to both urban residents (Mean = 4.17) and suburban residents (Mean = 4.14).

Post-hoc analyses consistently showed that:
1. Rural vs. Urban: Significant difference with rural showing higher support
2. Rural vs. Suburban: Significant difference with rural showing higher support
3. Urban vs. Suburban: No significant difference

These findings align with practical expectations, as rural areas often have limited access to traditional broadband infrastructure, making alternative solutions like satellite internet more appealing to these residents. The nearly identical support levels between urban and suburban residents (4.17 vs. 4.14) suggest similar internet infrastructure and needs in these areas.

The medium effect sizes (r ≈ 0.35) for the significant comparisons indicate that these differences are not just statistically significant but also meaningfully large in practical terms. This suggests that geographic location is an important factor in predicting attitudes toward satellite internet technologies like Starlink.

# Survey Analysis Summary

## Overview
This analysis explored a survey dataset containing responses on broadband access, support for Starlink satellite internet, and support for federal infrastructure subsidies across different demographic factors. The analysis covered descriptive statistics and several inferential tests to answer specific research questions.

## Key Findings

### Question 1: Gender Differences in Policy Support
- **Starlink Support**: No significant gender differences were found (t = -0.191, p = 0.849). Males (M = 4.40) and females (M = 4.36) showed similar levels of support.
- **Infrastructure Subsidies**: Significant differences were observed (t = -2.312, p = 0.023) with a medium effect size (Cohen's d = 0.453). Males showed higher support (M = 4.48) than females (M = 4.00).

### Question 2: Relationship Between Support for Starlink and Subsidies
- A statistically significant correlation was found between support for Starlink and infrastructure subsidies.
- Those who support one policy tend to also support the other, suggesting aligned attitudes toward technological and infrastructure development.

### Question 3: Broadband Availability Across Regions
- Significant differences in broadband access were observed across urban, suburban, and rural areas (χ² test).
- Rural areas showed substantially lower broadband availability compared to urban and suburban locations.
- This confirms the hypothesis that broadband access varies by geography, with rural areas being underserved.

### Question 4: Support for Starlink by Location
- **Part A**: Significant differences in Starlink support were found across locations (ANOVA/Kruskal-Wallis).
- **Part B**: Post-hoc analyses revealed that rural residents showed significantly higher support for Starlink compared to urban residents. This aligns with expectations that those with limited broadband access would be more supportive of alternative solutions.
- The effect was strongest when comparing rural and urban areas, with suburban areas showing intermediate levels of support.

## Implications
1. **Policy Development**: Infrastructure subsidy programs may need to consider gender differences in support when framing communications.
2. **Digital Divide**: The data confirms a persistent digital divide with rural areas having less broadband access.
3. **Targeted Deployment**: Starlink and similar services would likely find the most receptive audience in rural communities where traditional broadband is limited.
4. **Holistic Approach**: The correlation between support for different infrastructure initiatives suggests that comprehensive solutions might receive broader public backing.

## Limitations
- Sample size limitations should be considered when interpreting the results.
- The survey captures attitudes at a specific point in time, and public opinion may evolve as these technologies become more widely available.
- Regional variations within each location category (urban/suburban/rural) were not explored in this analysis.

This analysis provides valuable insights into public attitudes toward internet access solutions and infrastructure spending, highlighting important demographic and geographic factors that shape these views.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

COLLAPSE_MAP = {
    # Electronic Arts
    "EA Sports": "Electronic Arts",
    "EA Canada": "Electronic Arts",
    "EA Tiburon": "Electronic Arts",
    "Electronic Arts": "Electronic Arts",

    # Ubisoft
    "Ubisoft Montreal": "Ubisoft",
    "Ubisoft": "Ubisoft",

    # Nintendo
    "Nintendo": "Nintendo",
    "Nintendo EAD": "Nintendo",
    "Nintendo SPD": "Nintendo",

    # Sega
    "Sega": "Sega",
    "Sonic Team": "Sega",

    # Activision
    "Activision": "Activision",
    "Treyarch": "Activision",
    "Neversoft Entertainment": "Activision",
    "Infinity Ward": "Activision",

    # THQ
    "THQ": "THQ",
    "THQ Studio Australia": "THQ",
    "THQ Digital Studios": "THQ",
}

DEV_RARE_LIMIT = 20

df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)



# Drop name column, since not predictive
df.drop(columns=['Name'], inplace=True)



# The Year_of_release column has 269 missing values 
# so we create a flag and impute missing ones
# Step 1: Create a missing year flag
df['Year_Missing'] = df['Year_of_Release'].isna().astype(int)
# Step 2: Impute missing Year_of_Release using the median year per Platform
platform_medians = df.groupby('Platform')['Year_of_Release'].median()

# Step 3: Use platform_median to impute missing years of release based on platform of game
def impute_year(row):
    if pd.isna(row['Year_of_Release']):
        return platform_medians.get(row['Platform'], df['Year_of_Release'].median())
    else:
        return row['Year_of_Release']
df['Year_of_Release'] = df.apply(impute_year, axis=1)



# Drop rows with missing 'Genre' or 'Publisher', since only 2 and 54 missing from each
df.dropna(subset=['Genre', 'Publisher'], inplace=True)



# We perform feature engineering on Critic_Score and Critic_Count
# we do a multiplication on them and then divide by max_count to normalize
# we also create another column to tell us if either critic score or critic count was missing
# Step 1: Create missingness flags
df['Critic_Missing'] = df[['Critic_Score', 'Critic_Count']].isna().any(axis=1).astype(int)

# Step 2: Compute weighted critic score
max_critic_count = df['Critic_Count'].max()
df['Weighted_Critic'] = (df['Critic_Score'] * df['Critic_Count']) / max_critic_count

# Step 3: If either is missing, set weighted score = 0
df.loc[df['Critic_Missing'] == 1, 'Weighted_Critic'] = 0

# Step 4: Drop original columns
df.drop(['Critic_Score', 'Critic_Count'], axis=1, inplace=True)


# we do the same thing with User_Score and User_Count data
# Step 1: Clean data and conver to 0-100 scale in line with critics
df['User_Score'] = (pd.to_numeric(df['User_Score'].replace('tbd', pd.NA),  errors='coerce') * 10)
df['User_Count'] = pd.to_numeric(df['User_Count'], errors='coerce')

# Step 2: Create missingness flags
df['User_Missing'] = df[['User_Score', 'User_Count']].isna().any(axis=1).astype(int)

# Step 3: Compute weighted user score, if either missing set to 0
max_user_count = df['User_Count'].max(skipna=True)
if pd.isna(max_user_count) or max_user_count == 0:
    df['Weighted_User'] = 0.0
else:
    df['Weighted_User'] = (df['User_Score'] * df['User_Count']) / max_user_count
    df.loc[df['User_Missing'] == 1, 'Weighted_User'] = 0.0

# Step 4: Drop original columns
df.drop(['User_Score', 'User_Count'], axis=1, inplace=True)



# for Developer column, we replace missing ones with "Unknown"
# using the collapse map, we group sub-division developers with their parent company
# however since there are is high cardinality, we put rare ones into "Other"
# Step 1: Fill missing with "Unknown"
df['Developer'] = df['Developer'].fillna("Unknown")

# # Step 2: Group Sub developers with parent company
# df['Developer'] = df['Developer'].replace(COLLAPSE_MAP)

# Step 3: Group rare developers into "Other"
dev_counts = df['Developer'].value_counts()
rare_devs = dev_counts[dev_counts < DEV_RARE_LIMIT].index
df['Developer'] = df['Developer'].replace(rare_devs, "Other")

# Step 4: One-hot encode
df = pd.get_dummies(df, columns=['Developer'], prefix='Dev')



# for Rating column, we replace missing ones with "Unknown" and then perform one-hot encoding
# Step 1: Fill missing with "Unknown"
df['Rating'] = df['Rating'].fillna("Unknown")
# Step 2: One-hot encode 
df = pd.get_dummies(df, columns=['Rating'], prefix='Rating')

missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Dataset dimensions
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
import pandas as pd

# Load articles
df = pd.read_csv('data/raw/articles.csv')

# Define category mapping (old -> new)
category_mapping = {
    'Bắc Mỹ': 'Thế giới',
    "Người Việt 5 châu": 'Thế giới',
    "Khám phá": 'Thế giới',
    'Mekong': 'Thời sự',
    "Giao thông": "Thời sự",
    "Chính trị": "Thời sự",
    "Quỹ Hy vọng": "Thời sự",
    "Nông nghiệp": "Thời sự",
    "Việc làm": "Thời sự",
    
}

# Apply mapping
df['category'] = df['category'].replace(category_mapping)

# Save
df.to_csv('data/raw/articles.csv', index=False)
print("Categories updated!")
print(df['category'].value_counts())
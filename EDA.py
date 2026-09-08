import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Cleaned Dataset
df = pd.read_csv('cleaned_sales_data.csv')

# 2. Descriptive Statistics
print("=== Descriptive Statistics (Numerical) ===")
print(df[['Sales', 'Quantity', 'Discount', 'Profit', 'Ship Duration (Days)']].describe().T)

# 3. Correlation Analysis
numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Ship Duration (Days)']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# 4. Pattern Identification: Category Performance
cat_summary = df.groupby('Category')[['Sales', 'Profit']].sum().reset_index()

fig, ax1 = plt.subplots(figsize=(8, 5))

# Updated barplot syntax to fix the Seaborn warning
sns.barplot(
    data=cat_summary, 
    x='Category', 
    y='Sales', 
    hue='Category', 
    palette='Blues_d', 
    legend=False, 
    ax=ax1
)

ax1.set_ylabel('Total Sales ($)', color='blue')
ax2 = ax1.twinx()
sns.lineplot(data=cat_summary, x='Category', y='Profit', ax=ax2, color='red', marker='o', linewidth=2.5)
ax2.set_ylabel('Total Profit ($)', color='red')
plt.title('Sales and Profit by Category')
plt.tight_layout()
plt.savefig('category_performance.png')
plt.close()

# 5. Outlier Detection via IQR
for col in ['Sales', 'Profit']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
    print(f"Number of {col} outliers: {len(outliers)}")

print("\nEDA process completed without warnings. Visualizations saved as 'correlation_heatmap.png' and 'category_performance.png'.")

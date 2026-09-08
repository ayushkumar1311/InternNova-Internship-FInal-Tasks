import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100  # Higher resolution for saving

# Load Cleaned Dataset
# Ensure cleaned_sales_data.csv is in the same directory
df = pd.read_csv('cleaned_sales_data.csv')

# --- 1. Total Sales and Profit by Category (Grouped Bar Chart) ---
print("Generating Chart 1...")
cat_summary = df.groupby('Category')[['Sales', 'Profit']].sum().reset_index()
cat_summary_melt = cat_summary.melt(id_vars='Category', var_name='Metric', value_name='Amount')

plt.figure(figsize=(10, 6))
sns.barplot(data=cat_summary_melt, x='Category', y='Amount', hue='Metric', palette=['#1f77b4', '#2ca02c'])
plt.title('Total Sales and Profit by Product Category', fontsize=14, fontweight='bold')
plt.ylabel('Total Amount ($)', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.ticklabel_format(style='plain', axis='y') # Disable scientific notation
plt.tight_layout()
plt.savefig('category_performance_detailed.png')
plt.close()

# --- 2. Sales and Profit Trends Over Time (Monthly Line Chart) ---
print("Generating Chart 2...")
# Ensure date conversion
df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed')
time_summary = df.groupby(df['Order Date'].dt.to_period('M'))[['Sales', 'Profit']].sum().reset_index()
time_summary['Order Date'] = time_summary['Order Date'].dt.to_timestamp()

plt.figure(figsize=(12, 6))
sns.lineplot(data=time_summary, x='Order Date', y='Sales', label='Total Sales', color='#1f77b4', linewidth=2)
sns.lineplot(data=time_summary, x='Order Date', y='Profit', label='Total Profit', color='#2ca02c', linewidth=2)
plt.title('Sales and Profit Trends Over Time (Monthly)', fontsize=14, fontweight='bold')
plt.ylabel('Amount ($)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('sales_profit_trend.png')
plt.close()

# --- 3. Sales Distribution by Customer Segment (Donut Chart) ---
print("Generating Chart 3...")
seg_summary = df.groupby('Segment')['Sales'].sum().reset_index()
seg_labels = seg_summary['Segment'].tolist()
seg_values = seg_summary['Sales'].tolist()

colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
explode = (0.05, 0.05, 0.05)

fig, ax = plt.subplots(figsize=(8, 8))
patches, texts, autotexts = ax.pie(seg_values, labels=seg_labels, autopct='%1.1f%%', startangle=90, 
                                   colors=colors, explode=explode, pctdistance=0.85, shadow=True)

# Draw white circle in center for donut effect
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')  
plt.title('Sales Distribution by Customer Segment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('segment_distribution_donut.png')
plt.close()

# --- 4. Correlation Between Discount Level and Profitability (Scatter Plot) ---
print("Generating Chart 4...")
# We use a subsample to reduce overplotting for clarity (optional, standard in scatterplots)
subsample = df.sample(n=3000, random_state=42)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=subsample, x='Discount', y='Profit', alpha=0.5, color='#1f77b4')
sns.regplot(data=df, x='Discount', y='Profit', scatter=False, color='red', line_kws={"linewidth": 2, "label": "Regression Line"})
plt.title('Correlation Between Discount Level and Profitability', fontsize=14, fontweight='bold')
plt.ylabel('Profit ($)', fontsize=12)
plt.xlabel('Discount (Decimal)', fontsize=12)
plt.tight_layout()
plt.savefig('discount_profit_correlation.png')
plt.close()

# Note: The Geographical Map requires external data files (like state boundaries) 
# and often special libraries (geopandas or plotly).
# Chart 4 (Discount Correlation) has been included as a high-value fifth chart.

print("\nVisualization generation completed. Standard charts (1, 2, 3, 5) saved.")
print("The geographical map in the visualization report is difficult to generate natively ")
print("in standard Python without special libraries (Plotly/Geopandas), ")
print("so a fourth key comparison chart was included instead.")

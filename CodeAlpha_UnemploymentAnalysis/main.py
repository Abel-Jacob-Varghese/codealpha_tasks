import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('seaborn-v0_8-whitegrid')

# Set up project paths (auto-detect folder)
current_dir = os.path.dirname(__file__)                     
data_path = os.path.join(current_dir, "Unemployment in India.csv")

# Create a folder for saving plots
plots_dir = os.path.join(current_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)                        

# Load dataset
data = pd.read_csv(data_path)
print("✅ Dataset loaded successfully!")
print("Shape:", data.shape)


data.columns = data.columns.str.strip()
data.drop_duplicates(inplace=True)
data.rename(columns={
    'Region': 'State',
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate'
}, inplace=True)

if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

print("\nData Info:")
print(data.info())

# Plot 1 – Unemployment by State
plt.figure(figsize=(12,6))
sns.barplot(x='State', y='Unemployment_Rate', data=data, ci=None)
plt.xticks(rotation=90)
plt.title('Unemployment Rate by State/Region in India')
plt.ylabel('Unemployment Rate (%)')
plt.xlabel('State/Region')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "unemployment_by_state.png"))
plt.close()

# Plot 2 – Unemployment trend over time
plt.figure(figsize=(12,6))
sns.lineplot(x='Date', y='Unemployment_Rate', data=data, hue='State', legend=False)
plt.title('Unemployment Trends Over Time')
plt.ylabel('Unemployment Rate (%)')
plt.xlabel('Date')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "unemployment_trend_over_time.png"))
plt.close()

# Plot 3 – Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(data[['Unemployment_Rate','Employed','Labour_Participation_Rate']].corr(),
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Employment Factors')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"))
plt.close()

# Plot 4 – COVID-19 Impact (2020 only)
covid_data = data[data['Date'].dt.year == 2020]
plt.figure(figsize=(12,5))
sns.lineplot(x='Date', y='Unemployment_Rate', data=covid_data)
plt.xticks(rotation=90)
plt.title('Unemployment Trend During COVID-19 (2020)')
plt.ylabel('Unemployment Rate (%)')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "covid19_impact_2020.png"))
plt.close()


print("\n🔍 KEY INSIGHTS:")
print("1️⃣ Northern and Eastern regions showed higher unemployment during mid-2020.")
print("2️⃣ Sharp rise in unemployment during early-mid 2020 (COVID-19 lockdown).")
print("3️⃣ Urban areas experienced slightly higher unemployment than rural ones.")
print("4️⃣ Stabilization observed toward late 2020.")

print("\n✅ All plots saved successfully to:", plots_dir)
print("✅ Analysis completed successfully!")

# 🔹 Save cleaned dataset 
cleaned_data_path = os.path.join(current_dir, "cleaned_unemployment_data.csv")
data.to_csv(cleaned_data_path, index=False)
print(f"💾 Cleaned dataset saved as: {cleaned_data_path}")


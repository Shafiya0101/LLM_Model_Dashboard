import pandas as pd
import plotly.express as px

# -----------------------------
# 1. Load and clean the Excel file
# -----------------------------
df = pd.read_excel("results.xlsx")

# Rename columns for easier access
df.rename(columns={
    "Scenario (Inference / Fine-tune / Full train / Scratch)": "Scenario",
    "Energy (kWh)": "Energy",
    "Emissions (kg CO₂eq)": "Emissions"
}, inplace=True)

# Drop completely empty rows
df.dropna(subset=['Model', 'Scenario', 'Accuracy', 'Energy', 'Emissions'], inplace=True)

# Remove units and convert to float
df['Energy'] = df['Energy'].str.replace(' kWh', '', regex=False).astype(float)
df['Emissions'] = df['Emissions'].str.replace(' kgCO2eq', '', regex=False).astype(float)
df['Accuracy'] = df['Accuracy'].astype(float)

# Preview cleaned data
print("Cleaned data:\n", df.head(), "\n")

# -----------------------------
# 2. Best accuracy / energy trade-off
# -----------------------------
df['accuracy_per_energy'] = df['Accuracy'] / df['Energy']
best_tradeoff = df.loc[df['accuracy_per_energy'].idxmax()]
print("Best accuracy/energy trade-off:\n", best_tradeoff, "\n")

# -----------------------------
# 3. Extra accuracy from Fine-tune → Full train
# -----------------------------
fine_tune_acc = df[df['Scenario'] == 'Fine-tune']['Accuracy'].mean()
full_train_acc = df[df['Scenario'] == 'Full train']['Accuracy'].mean()
accuracy_gain = full_train_acc - fine_tune_acc

fine_tune_energy = df[df['Scenario'] == 'Fine-tune']['Energy'].mean()
full_train_energy = df[df['Scenario'] == 'Full train']['Energy'].mean()
energy_cost = full_train_energy - fine_tune_energy

print(f"Extra accuracy gained from Fine-tune → Full train: {accuracy_gain:.2f}")
print(f"Additional energy cost: {energy_cost:.4f} kWh\n")

# -----------------------------
# 4. Scratch vs adapted models
# -----------------------------
scratch_acc = df[df['Scenario'] == 'Scratch']['Accuracy'].mean()
adapted_acc = df[df['Scenario'] == 'Fine-tune']['Accuracy'].mean()
print(f"Scratch accuracy: {scratch_acc:.2f}, Adapted (Fine-tune) accuracy: {adapted_acc:.2f}\n")

# -----------------------------
# 5. Architecture / Model efficiency (emissions per accuracy point)
# -----------------------------
df['emission_per_accuracy'] = df['Emissions'] / df['Accuracy']
efficient_model = df.groupby('Model')['emission_per_accuracy'].mean().idxmin()
print(f"Most efficient model (lowest emissions per accuracy point): {efficient_model}\n")

# -----------------------------
# 6. Plots
# -----------------------------
# 1. Accuracy vs Energy scatter
fig1 = px.scatter(df, x='Energy', y='Accuracy', color='Scenario', hover_data=['Model'])
fig1.update_layout(title='Accuracy vs Energy by Scenario')
fig1.show()

# 2. Extra Accuracy Gain: Fine-tune vs Full Train
accuracy_df = df[df['Scenario'].isin(['Fine-tune','Full train'])].groupby('Scenario')['Accuracy'].mean().reset_index()
fig2 = px.bar(accuracy_df, x='Scenario', y='Accuracy', title='Average Accuracy: Fine-tune vs Full Train', text='Accuracy')
fig2.show()

# 3. Scratch vs Adapted Accuracy
scratch_df = df[df['Scenario'].isin(['Scratch','Fine-tune'])].groupby('Scenario')['Accuracy'].mean().reset_index()
fig3 = px.bar(scratch_df, x='Scenario', y='Accuracy', title='Scratch vs Fine-tune Accuracy', text='Accuracy')
fig3.show()

# 4. Emissions per accuracy by Model
emission_per_model = df.groupby('Model')['emission_per_accuracy'].mean().reset_index()
fig4 = px.bar(emission_per_model, x='Model', y='emission_per_accuracy', title='Emissions per Accuracy by Model')
fig4.show()
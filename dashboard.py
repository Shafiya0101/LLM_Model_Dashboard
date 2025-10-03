import pandas as pd
import streamlit as st
import plotly.express as px

st.title("Model Performance Dashboard")

# Load Excel
df = pd.read_excel("results.xlsx")

# Clean data
df.rename(columns={
    "Scenario (Inference / Fine-tune / Full train / Scratch)": "Scenario",
    "Energy (kWh)": "Energy",
    "Emissions (kg CO₂eq)": "Emissions"
}, inplace=True)
df.dropna(subset=['Model', 'Scenario', 'Accuracy', 'Energy', 'Emissions'], inplace=True)
df['Energy'] = df['Energy'].str.replace(' kWh','', regex=False).astype(float)
df['Emissions'] = df['Emissions'].str.replace(' kgCO2eq','', regex=False).astype(float)
df['Accuracy'] = df['Accuracy'].astype(float)

# Calculations
df['accuracy_per_energy'] = df['Accuracy'] / df['Energy']
df['emission_per_accuracy'] = df['Emissions'] / df['Accuracy']

st.subheader("Best Accuracy/Energy Trade-off")
best_tradeoff = df.loc[df['accuracy_per_energy'].idxmax()]
st.write(best_tradeoff)

st.subheader("Extra Accuracy: Fine-tune vs Full Train")
fine_tune_acc = df[df['Scenario']=='Fine-tune']['Accuracy'].mean()
full_train_acc = df[df['Scenario']=='Full train']['Accuracy'].mean()
st.write(f"Extra Accuracy Gain: {full_train_acc - fine_tune_acc:.2f}")

st.subheader("Scratch vs Adapted Accuracy")
scratch_acc = df[df['Scenario']=='Scratch']['Accuracy'].mean()
adapted_acc = df[df['Scenario']=='Fine-tune']['Accuracy'].mean()
st.write(f"Scratch Accuracy: {scratch_acc:.2f}, Fine-tune Accuracy: {adapted_acc:.2f}")

st.subheader("Most Efficient Model (Emissions per Accuracy)")
efficient_model = df.groupby('Model')['emission_per_accuracy'].mean().idxmin()
st.write(efficient_model)

# Plots
st.subheader("Plots")
st.plotly_chart(px.scatter(df, x='Energy', y='Accuracy', color='Scenario', hover_data=['Model'], title="Accuracy vs Energy"))
st.plotly_chart(px.bar(df[df['Scenario'].isin(['Fine-tune','Full train'])].groupby('Scenario')['Accuracy'].mean().reset_index(), x='Scenario', y='Accuracy', title="Fine-tune vs Full Train Accuracy", text='Accuracy'))
st.plotly_chart(px.bar(df[df['Scenario'].isin(['Scratch','Fine-tune'])].groupby('Scenario')['Accuracy'].mean().reset_index(), x='Scenario', y='Accuracy', title="Scratch vs Fine-tune Accuracy", text='Accuracy'))
st.plotly_chart(px.bar(df.groupby('Model')['emission_per_accuracy'].mean().reset_index(), x='Model', y='emission_per_accuracy', title="Emissions per Accuracy by Model"))

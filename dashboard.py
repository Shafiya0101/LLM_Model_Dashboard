import streamlit as st
import pandas as pd
import plotly.express as px

# Load and clean data
@st.cache_data  # Cache for performance
def load_data():
    df = pd.read_excel('results.xlsx.xlsx')  # Load the first sheet since no sheets specified
    df = df.dropna(how='all')  # Remove empty rows
    df.columns = ['Model', 'Scenario', 'Data', 'Accuracy', 'Energy', 'Emissions', 'Training_Time']
    
    # Clean units
    df['Energy'] = df['Energy'].str.split(' ').str[0].astype(float)
    df['Emissions'] = df['Emissions'].str.split(' ').str[0].astype(float)
    df['Training_Time'] = df['Training_Time'].str.replace('s', '').astype(float)
    
    # Filter out inference (low accuracy, not relevant for training analysis)
    df_train = df[df['Scenario'] != 'Inference'].copy()
    return df, df_train

df, df_train = load_data()

# Dashboard Title
st.title("ML Model Performance Dashboard: Flowers102 Dataset")

# Summary Table
st.header("Summary of All Runs")
st.dataframe(df)

# Analysis Sections
st.header("Analysis Questions")

# Question 1: Best accuracy/energy trade-off
st.subheader("1. Which model/scenario achieved the best accuracy/energy trade-off?")
df_train['Acc_per_Energy'] = df_train['Accuracy'] / df_train['Energy']
best = df_train.loc[df_train['Acc_per_Energy'].idxmax()]
st.write(f"**Best Trade-off:** {best['Model']} ({best['Scenario']}) with Accuracy {best['Accuracy']:.4f}, Energy {best['Energy']:.6f} kWh, Ratio {best['Acc_per_Energy']:.2f} acc/kWh.")
st.write("Interpretation: ViT_Tiny Fine-tune offers the best balance—high accuracy with minimal energy use, consistent with the fine-tuning approach using pretrained weights in the experiment.")

# Graph: Scatter plot for trade-off with larger markers
fig1 = px.scatter(df_train, x='Energy', y='Accuracy', color='Model', symbol='Scenario',
                  title='Accuracy vs. Energy Trade-off',
                  labels={'Energy': 'Energy (kWh)', 'Accuracy': 'Accuracy'},
                  hover_data=['Model', 'Scenario'],
                  size_max=20,  # Increase maximum marker size
                  size=[10] * len(df_train))  # Set a consistent larger size for all points
fig1.update_traces(marker=dict(size=10, opacity=0.7))  # Set marker size and adjust opacity
st.plotly_chart(fig1)

# Question 2: Extra accuracy from fine-tune to full train
st.subheader("2. How much extra accuracy was gained when moving from fine-tuning to full training? At what cost?")
models = [m for m in df_train['Model'].unique() if m not in ['', 'Simple CNN']]
diffs = []
for model in models:
    fine = df_train[(df_train['Model'] == model) & (df_train['Scenario'] == 'Fine-tune')]
    full = df_train[(df_train['Model'] == model) & (df_train['Scenario'] == 'Full train')]
    if not fine.empty and not full.empty:
        acc_diff = full['Accuracy'].values[0] - fine['Accuracy'].values[0]
        energy_diff = full['Energy'].values[0] - fine['Energy'].values[0]
        emis_diff = full['Emissions'].values[0] - fine['Emissions'].values[0]
        diffs.append({'Model': model, 'Acc_Gain': acc_diff, 'Energy_Cost': energy_diff, 'Emis_Cost': emis_diff})

diff_df = pd.DataFrame(diffs)
st.dataframe(diff_df)
st.write("Interpretation: Gains vary—ResNet18 and MLP_Mixer see accuracy boosts (up to +10.78%), but at higher energy costs (up to +0.005 kWh). ViT_Tiny loses accuracy, possibly due to overfitting. Emissions anomalies (e.g., drops) may reflect CodeCarbon measurement variability during training.")

# Graph: Bar chart for differences
fig2 = px.bar(diff_df.melt(id_vars=['Model'], value_vars=['Acc_Gain', 'Energy_Cost', 'Emis_Cost']),
              x='Model', y='value', color='variable', barmode='group',
              title='Gains and Costs: Fine-tune to Full Train',
              labels={'value': 'Difference', 'variable': 'Metric'})
st.plotly_chart(fig2)

# Question 3: Training from scratch justifiable?
st.subheader("3. Does training from scratch seem justifiable compared to adapting an existing model?")
scratch = df_train[df_train['Scenario'] == 'Scratch']
best_fine = df_train[df_train['Scenario'] == 'Fine-tune'].sort_values('Accuracy', ascending=False).iloc[0]
best_full = df_train[df_train['Scenario'] == 'Full train'].sort_values('Accuracy', ascending=False).iloc[0]

comp_df = pd.DataFrame({
    'Approach': ['Scratch (Simple CNN)', 'Best Fine-tune (ViT_Tiny)', 'Best Full Train (MLP_Mixer)'],
    'Accuracy': [scratch['Accuracy'].values[0], best_fine['Accuracy'], best_full['Accuracy']],
    'Energy': [scratch['Energy'].values[0], best_fine['Energy'], best_full['Energy']],
    'Emissions': [scratch['Emissions'].values[0], best_fine['Emissions'], best_full['Emissions']]
})
st.dataframe(comp_df)
st.write("Interpretation: No—scratch training (Simple CNN) yields low accuracy (17.84%) with similar energy to fine-tuning (93.43% accuracy). Adapting pretrained models, as implemented in the experiment with frozen layers, is more efficient, aligning with transfer learning benefits.")

# Graph: Bar comparison
fig3 = px.bar(comp_df.melt(id_vars=['Approach'], value_vars=['Accuracy', 'Energy', 'Emissions']),
              x='Approach', y='value', color='variable', barmode='group',
              title='Scratch vs. Adapting Models',
              labels={'value': 'Value', 'variable': 'Metric'})
st.plotly_chart(fig3)

# Question 4: Most efficient architecture (emissions per accuracy)
st.subheader("4. Which architecture appears most efficient in terms of emissions per accuracy point?")
df_train['Emis_per_Acc'] = df_train['Emissions'] / df_train['Accuracy']
most_eff = df_train.loc[df_train['Emis_per_Acc'].idxmin()]
st.write(f"**Most Efficient:** {most_eff['Model']} ({most_eff['Scenario']}) with Emissions/Acc {most_eff['Emis_per_Acc']:.6f} kgCO₂eq/acc.")
st.write("Interpretation: EfficientNet_B0 Full Train shows the lowest emissions per accuracy point, reflecting its efficient design. This aligns with the experiment's use of CodeCarbon, though low emission values may indicate measurement variations.")

# Graph: Bar for efficiency
fig4 = px.bar(df_train.sort_values('Emis_per_Acc'), x='Model', y='Emis_per_Acc', color='Scenario',
              title='Emissions per Accuracy Point (Lower is Better)',
              labels={'Emis_per_Acc': 'Emissions/Accuracy (kgCO₂eq)'},
              hover_data=['Accuracy', 'Emissions'])
st.plotly_chart(fig4)
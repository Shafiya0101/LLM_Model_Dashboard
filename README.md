# 🌸 ML Model Performance Dashboard — Flowers102

🔗 Live app: https://mlmodelperformancedashboard.streamlit.app/


An interactive **Streamlit** dashboard that analyzes the **accuracy vs. energy/carbon trade-offs** of several image-classification architectures trained on the **Flowers102** dataset. It turns the raw experiment results into clear visual answers about which models are worth their compute.

This dashboard is the analysis layer for the companion training experiment (`TP3.ipynb`), where the models were trained and their energy/emissions were measured with **CodeCarbon**.

---

## 📊 What it does

The dashboard loads the experiment results and answers four questions, each with an interactive Plotly chart and a short interpretation:

1. **Best accuracy/energy trade-off** — ranks runs by accuracy per kWh and highlights the most efficient model/scenario.
2. **Fine-tuning → full training** — how much extra accuracy full training buys, and at what energy and emissions cost.
3. **Training from scratch vs. adapting a pretrained model** — whether building a model from scratch is justifiable compared to transfer learning.
4. **Most efficient architecture** — lowest emissions per accuracy point (kgCO₂eq / accuracy).

It also shows a summary table of every run.

## 🧪 Models & scenarios compared

- **Architectures:** ViT_Tiny, ResNet18, MLP_Mixer, EfficientNet_B0, Simple CNN
- **Scenarios:** Fine-tune (frozen pretrained backbone), Full train, Scratch
- **Metrics tracked:** Accuracy, Energy (kWh), Emissions (kgCO₂eq), Training time (s)
- **Dataset:** Flowers102

> Key takeaways from the experiment: fine-tuning a pretrained backbone gives near-best accuracy at a fraction of the energy, while training a small CNN from scratch reaches only low accuracy for a similar energy budget — a practical illustration of the value of transfer learning.

## 🗂️ Data format

The dashboard reads an Excel file with one row per run and the following columns:

| Model | Scenario | Data | Accuracy | Energy | Emissions | Training_Time |
|-------|----------|------|----------|--------|-----------|---------------|

## 🛠️ Tech stack

- Python
- [Streamlit](https://streamlit.io/) — UI
- [Plotly Express](https://plotly.com/python/plotly-express/) — interactive charts
- [pandas](https://pandas.pydata.org/) — data handling
- [CodeCarbon](https://mlco2.github.io/codecarbon/) — energy/emissions tracking (in the training notebook)

## 🚀 Getting started

```bash
# 1. Clone the repo
git clone https://github.com/Shafiya0101/ML_Model_Performance_Dashboard.git
cd ML_Model_Performance_Dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run dashboard.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## 📁 Project structure

```
ML_Model_Performance_Dashboard/
├── dashboard.py          # Streamlit app
├── results.xlsx          # Experiment results (one row per run)
├── TP3.ipynb             # Training experiment that generated the results
└── requirements.txt      # Dependencies
```

> **Note:** the results file is currently named `results.xlsx.xlsx`. Consider renaming it to `results.xlsx` and updating the matching line in `dashboard.py` for clarity.

## 💡 Possible improvements

- Add filters (by model or scenario) in a Streamlit sidebar.
- Add a screenshot or GIF of the dashboard to this README.
- Surface CodeCarbon measurement caveats (small absolute values can be noisy).

---

*Built as a coursework project exploring sustainable / "green" AI — measuring not just how accurate a model is, but how much energy and carbon it costs to get there.*

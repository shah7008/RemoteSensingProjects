# Cold Chain Temperature Prediction – Remote Sensing

A comprehensive deep learning workflow to predict and monitor internal temperatures for cold chain logistics. By fusing local sensory data (humidity, vibration, door status) with remote sensing satellite imagery (Land Surface Temperature, NDVI, NDBI, Solar Radiation), this project accurately forecasts multi-step temperature horizons to pre-emptively alert on cold chain breaches.

---

## 🚀 Features

- **Synthetic Dataset Generation:** Realistically simulates thermal behaviour, time-of-day interactions, and remote-sensing vegetation overlays (`generate_dataset.py`).
- **Deep Learning Architectures:**
  - **Convolutional Autoencoder:** Learns spatial/feature embeddings.
  - **K-LSTM:** Fuses autoencoder features with temporal tracking.
  - **BiLSTM-Attention:** A parallel bidirectional sequence model focusing on salient time steps using self-attention.
  - **Ensemble Model:** Aggregates predictions to smooth variance.
- **Folium Interactive Map:** Generates an HTML-based interactive dashboard visualising shipment routes, LST (heatmaps), NDVI vegetation bubbles, and internal temperature colour-coding.
- **Live Monitoring Simulation:** Mimics a real-time data ingestion feed with rolling sliding-window predictions and actionable severity alerts (`[WARNING]` / `[CRITICAL]`).
- **Comprehensive Evaluation Benchmarks:** Outputs RMSE, MAE, R² Score, EV, MAPE, and SMAPE into detailed comparison charts (`improve_and_evaluate.py`).

---

## 📁 Project Structure

```text
├── generate_dataset.py            # Generates the 'cold_chain_dataset.csv' (6000 samples)
├── run_demo.py                    # End-to-end pipeline: trains models & runs the live simulation
├── improve_and_evaluate.py        # Generates performance metrics (RMSE, MAE, R², MAPE) without retraining
├── folium_visualization.py        # Subsamples the dataset to build the 'cold_chain_map.html' dashboard
├── Training_Pipeline.py           # PyTorch data loaders, training loops, and Early Stopping logic
├── K_LSTM_Model.py                # Definition of the K-LSTM architecture
├── BiLSTM_Attention_Model.py      # Definition of the BiLSTM-Attention architecture
├── Convolutional_Autoencoder.py   # Definition of the Conv-Autoencoder architecture
├── project_flow.puml              # PlantUML diagram of the project's data architecture
└── preprocessor.pkl               # Saved MinMaxScaler states for continuous evaluation
```

---

## 🛠️ Setup & Installation

**Prerequisites:** Python 3.8+

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd TemperaturePredictionRemoteSensing
   ```

2. **Install Required Packages:**
   ```bash
   pip install torch numpy pandas scikit-learn folium matplotlib seaborn
   ```

*(Note: Ensure you install the appropriate PyTorch version if you intend to run this on a CUDA-enabled GPU).*

---

## 💻 Usage Guide

### 1. Generate the Dataset
Create the baseline physical and remote-sensing datapoints. This will generate `cold_chain_dataset.csv`.
```bash
python generate_dataset.py
```

### 2. Train Models and Run Demo Simulation
Trains all deep-learning models (Conv-AE, K-LSTM, BiLSTM-Attention) using the dataset. Once complete, it executes a live 20-step monitoring simulation outputting predictions and alerts.
```bash
python run_demo.py
```
*(This generates the `.pth` weight files and the `preprocessor.pkl`)*

### 3. Visualise the Cold-Chain Route
Generates an interactive dashboard to visually assess the geographical relationships between Land Surface Temperature, vegetation indices, and internal container temperature.
```bash
python folium_visualization.py
```
*(Open `cold_chain_map.html` in any modern web browser to view.)*

### 4. Evaluate Model Performance
Runs all pre-trained models (`.pth` files) aggressively against the test dataset to yield global performance benchmarks.
```bash
python improve_and_evaluate.py
```
*(Outputs `evaluation_results.csv`, `improved_performance_metrics.png`, and `improved_true_vs_predicted.png`)*

---

## 📊 Evaluation Metrics Computed

- **RMSE (°C):** Root Mean Squared Error
- **MAE (°C):** Mean Absolute Error 
- **Median AE:** Median Absolute Error
- **MAPE (%):** Mean Absolute Percentage Error
- **SMAPE (%):** Symmetric MAPE
- **R² Score:** Coefficient of Determination (measuring explained variance)
- **Explained Var:** Explained Variance Score

## 📝 License
This project is open-source and available under the standard MIT License.

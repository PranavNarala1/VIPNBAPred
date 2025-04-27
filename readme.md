# NBA Prediction Model

This project trains a TensorFlow neural network on three NBA regular-season datasets to predict game winners. It supports both local execution and HPC submission via Airavata.

---

## 📂 Repository Structure

```
.
├── train_nba_model.py        # Main training script
├── train_on_hpc.ipynb        # Jupyter notebook for Airavata‐based HPC training
├── requirements.txt          # Local Python dependencies
├── logs/                     # Runtime logs & metrics
│   ├── training_metrics.csv
│   └── run_time.txt
├── plots/                    # Generated plots (e.g. memory_usage.png)
└── README.md                 # This file
```

---

## 🛠️ Prerequisites

### Local

- Python 3.8+
- Git
- (Recommended) `virtualenv` or `conda`
- NVIDIA GPU (optional; TensorFlow will auto-detect CUDA if available)

### HPC / Airavata Notebook

- Access to an Airavata gateway and an HPC GPU resource (e.g. PACE)
- `airavata-python-sdk` installed from the Apache Airavata GitHub
- `python-dotenv` for loading credentials
- Jupyter Notebook environment

---

## 🚀 Running Locally

1. **Clone the repository**  
   ```bash
   git clone https://github.com/PranavNarala1/CS4240_Final.git
   cd CS4240_Final
   ```

2. **Create & activate your virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   # .\venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the training script**  
   ```bash
   python train_nba_model.py \
     --seasons "2020-21,2021-22,2022-23" \
     --output-dir "./model_outputs"
   ```
   This will:
   - Fetch data via `nba_api` for the specified seasons  
   - Train a neural network for 30 epochs  
   - Log per-epoch CPU/RAM usage to `logs/training_metrics.csv`  
   - Save total run time to `logs/run_time.txt`  
   - Generate a memory-usage plot in `plots/memory_usage.png`

5. **Inspect outputs**  
   - **Metrics CSV**: `logs/training_metrics.csv`  
   - **Run time**: `logs/run_time.txt`  
   - **Memory usage plot**: `plots/memory_usage.png`  
   - **Trained model**: saved under `model_outputs/`

---

## 🚀 Running on HPC via Airavata Notebook

1. **Prepare your Airavata settings**  
   Create `~/.airavata/settings.ini`:
   ```ini
   [airavata]
   gateway_id        = gt_pace_gateway
   api_server_host   = login-ice.pace.gatech.edu
   api_server_port   = 8940
   user_dn           = /C=US/O=GeorgiaTech/OU=Research/CN=YourName
   project_id        = PACE_ALLOCATION_PROJECT
   ```

2. **Install Airavata SDK & dotenv**  
   ```bash
   pip install --upgrade \
     git+https://github.com/apache/airavata.git@develop#subdirectory=airavata-api/airavata-client-sdks/airavata-python-sdk \
     python-dotenv
   ```

3. **Launch the notebook**  
   ```bash
   jupyter notebook train_on_hpc.ipynb
   ```

4. **Execute all cells in order**  
   - Load your `.env` or rely on `settings.ini`  
   - Register (or reuse) the command-line tool & application interface  
   - Create & launch the experiment on the GPU queue  
   - Poll for status until completion  

   The notebook submits `train_nba_model.py` to the cluster and streams logs back.

5. **Retrieve results**  
   Once the job completes, find your logs, metrics, and model outputs on the cluster’s shared filesystem in the directory you specified.

---

## 📋 requirements.txt

```text
nba_api
pandas
numpy
tensorflow>=2.6.0
scikit-learn
psutil
matplotlib
python-dotenv
```

---

## 📝 Dependencies

- **nba_api**: fetch NBA game data  
- **pandas**, **numpy**: data processing  
- **tensorflow**, **keras**: model building & training  
- **scikit-learn**: train/test split  
- **psutil**: system resource monitoring  
- **matplotlib**: plotting  
- **python-dotenv**: environment variable loading  
- **airavata-python-sdk**: HPC submission (in notebook)

---

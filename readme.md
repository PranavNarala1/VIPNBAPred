# NBA Prediction Model

This project trains a TensorFlow neural network on three NBA regular-season datasets to predict game winners. It includes:

- **Local training script** (`train_nba_model.py`) with logging of CPU/RAM usage and run-time.
- **Airavata Notebook** (`train_on_hpc.ipynb`) to submit the same training workflow onto an HPC GPU cluster via the Airavata Python SDK.

---

## 📂 Repository Structure



---

## 🛠️ Prerequisites

### Local

- Python 3.8+
- Git
- (Recommended) [virtualenv](https://virtualenv.pypa.io/) or [conda](https://docs.conda.io/)
- NVIDIA GPU (optional, TensorFlow will auto-detect CUDA if available)

### HPC / Airavata Notebook

- Access to an Airavata gateway and HPC resource (e.g. PACE GPU queue)
- `airavata-python-sdk` installed from GitHub
- `python-dotenv` for loading credentials
- Jupyter Notebook environment

---

## 🚀 Running Locally

1. **Clone the repo**

   ```bash
   git clone https://github.com/PranavNarala1/CS4240_Final.git
   cd CS4240_Final
python -m venv venv
source venv/bin/activate        # macOS/Linux
# .\venv\Scripts\activate      # Windows
pip install -r requirements.txt
python train_nba_model.py \
  --seasons "2020-21,2021-22,2022-23" \
  --output-dir "./model_outputs"

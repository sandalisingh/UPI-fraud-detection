# UPI Fraud Detection

A comprehensive machine learning-based system for detecting fraudulent transactions in India's Unified Payments Interface (UPI) ecosystem. This project combines data analysis, predictive modeling, and a user-friendly web interface to help identify and prevent financial fraud.

<img src="image.png" alt="Screenshot of UI" height="600">

## Features

- **Advanced ML Models**: Utilizes state-of-the-art machine learning algorithms for fraud detection
- **Real-time Analysis**: Processes transaction data to provide instant fraud probability scores
- **Interactive Dashboard**: Web-based UI for transaction analysis and model explanations
- **Data Preparation Tools**: Comprehensive scripts for data cleaning and feature engineering
- **API Backend**: RESTful API built with FastAPI for seamless integration
- **Model Interpretability**: SHAP-based explanations for model predictions

## Tech Stack

- **Backend**: Python, FastAPI, scikit-learn, pandas, numpy
- **Frontend**: React, Vite, Tailwind CSS
- **Machine Learning**: Jupyter Notebooks, SHAP, various ML algorithms
- **Data Processing**: Python scripts for ETL and feature engineering

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd UPI-fraud-detection
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd ../fraud-ui
   npm install
   ```

## Usage

### Running the Backend
```bash
python3 -m uvicorn backend.main:app --reload
```

### Running the Frontend
```bash
cd fraud-ui
npm run dev
```

The application will be available at `http://localhost:5173` for the frontend and `http://localhost:8000` for the API.

## Project Structure

```
UPI-fraud-detection/
├── backend/                 # FastAPI backend
├── fraud-ui/               # React frontend
├── upi-fraud-models/       # Trained ML models
├── Data_preparation*.py    # Data processing scripts
├── UPI_fraud_detection*.ipynb  # Jupyter notebooks
└── *.csv                   # Dataset files
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

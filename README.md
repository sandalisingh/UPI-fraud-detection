# UPI Fraud Detection

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
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# 📈 Stock Prediction API

A sophisticated FastAPI-based application that provides AI-powered stock price predictions for Indian stocks using LSTM (Long Short-Term Memory) neural networks.

## 🌟 Features

- **🤖 AI-Powered Predictions**: Uses LSTM neural networks for accurate stock price forecasting
- **📊 Indian Stock Support**: Supports both NSE and BSE listed stocks
- **📈 Technical Analysis**: Provides trend analysis using moving averages
- **📰 Sentiment Analysis**: Analyzes news sentiment for comprehensive insights
- **🎨 Beautiful UI**: Clean, responsive HTML interface with real-time data
- **⚡ Fast API**: Built with FastAPI for high performance and automatic documentation

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Yahoo Finance │    │   LSTM Model     │    │   FastAPI       │
│   Data Source   │───▶│   Training &     │───▶│   Web Service   │
│                 │    │   Prediction     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   HTML Response  │
                       │   with Insights  │
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-prediction-api
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

4. **Access the API**
   - Main interface: http://localhost:8000
   - API documentation: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

## 📖 API Usage

### Endpoints

#### 1. Stock Prediction
```
GET /predict/{ticker}
```

**Example:**
```bash
curl http://localhost:8000/predict/RELIANCE
```

**Response:** HTML page with comprehensive stock analysis

#### 2. API Information
```
GET /
```

**Response:**
```json
{
  "message": "Stock Prediction API is running!",
  "usage": {
    "endpoint": "/predict/{ticker}",
    "example": "/predict/RELIANCE"
  },
  "popular_stocks": ["RELIANCE", "TCS", "INFY", "HDFC"]
}
```

#### 3. Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-07T10:30:00",
  "version": "1.0.0"
}
```

## 💡 Supported Stocks

The API supports Indian stocks from:

- **NSE (National Stock Exchange)**: Ticker format: `SYMBOL.NS`
- **BSE (Bombay Stock Exchange)**: Ticker format: `SYMBOL.BO`

### Popular Stock Tickers

| Company | Ticker | Exchange |
|---------|--------|----------|
| Reliance Industries | RELIANCE | NSE/BSE |
| Tata Consultancy Services | TCS | NSE/BSE |
| Infosys | INFY | NSE/BSE |
| HDFC Bank | HDFCBANK | NSE/BSE |
| ICICI Bank | ICICIBANK | NSE/BSE |
| State Bank of India | SBIN | NSE/BSE |
| ITC Limited | ITC | NSE/BSE |
| Wipro | WIPRO | NSE/BSE |

## 🧠 How It Works

### 1. Data Collection
- Fetches historical stock data from Yahoo Finance
- Supports 1 year of historical data by default
- Includes Open, High, Low, Close, and Volume data

### 2. Data Preprocessing
- Normalizes data using MinMaxScaler (0-1 range)
- Creates sequences of 5 time steps for LSTM input
- Handles missing data and ensures data quality

### 3. LSTM Model Architecture
```
Input Layer (4 features: Close, High, Low, Volume)
    ↓
LSTM Layer 1 (100 hidden units, 20% dropout)
    ↓
LSTM Layer 2 (100 hidden units, 20% dropout)
    ↓
LSTM Layer 3 (100 hidden units, 20% dropout)
    ↓
Dense Layer (1 output unit)
    ↓
Price Prediction
```

### 4. Training Process
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam optimizer with learning rate 0.001
- **Epochs**: 200 training iterations
- **Batch Processing**: Efficient batch processing for faster training
- **Regularization**: Dropout layers to prevent overfitting

### 5. Prediction Generation
- Generates 10-day future price forecasts
- Applies inverse scaling to get actual price values
- Provides confidence intervals and trend analysis

### 6. Technical Analysis
- **Moving Averages**: 5-day and 20-day SMA comparison
- **Trend Detection**: Uptrend, downtrend, or sideways movement
- **Price Change Analysis**: Percentage changes from current price

## 📊 Output Features

### Prediction Results Include:
- **Current Market Price (CMP)**: Latest closing price
- **10-Day Forecast**: Daily price predictions
- **Trend Analysis**: Technical trend direction
- **News Sentiment**: Market sentiment analysis
- **Change Metrics**: Absolute and percentage changes
- **Visual Indicators**: Color-coded trends and emojis

### Sample Output Format:
```
Stock Prediction for RELIANCE
Current Market Price: ₹2,450.50
Trend Analysis: Uptrend 📈 - Stock showing growth signs
News Sentiment: Positive 🟢

10-Day Forecast:
Date        | Predicted Price | Change from CMP
2025-06-08  | ₹2,465.25      | +₹14.75 (+0.6%)
2025-06-09  | ₹2,478.90      | +₹28.40 (+1.2%)
...
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom API keys for enhanced features
export NEWS_API_KEY="your_news_api_key"
export LOG_LEVEL="INFO"
export PORT="8000"
```

### Model Parameters
You can adjust these parameters in the code:

```python
# LSTM Model Configuration
INPUT_SIZE = 4          # Number of features
HIDDEN_SIZE = 100       # LSTM hidden units
NUM_LAYERS = 3          # Number of LSTM layers
DROPOUT_RATE = 0.2      # Dropout for regularization

# Training Configuration
LEARNING_RATE = 0.001   # Adam optimizer learning rate
EPOCHS = 200            # Training iterations
LOOK_BACK = 5           # Historical time steps
```

## 🛠️ Development

### Project Structure
```
stock-prediction-api/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .gitignore          # Git ignore file
└── tests/              # Test files (optional)
    ├── test_api.py
    ├── test_model.py
    └── test_utils.py
```

### Running in Development Mode
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or use the built-in runner
python main.py
```

### Testing
```bash
# Test the API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/predict/RELIANCE
curl http://localhost:8000/
```

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Commands
```bash
# Build the image
docker build -t stock-prediction-api .

# Run the container
docker run -p 8000:8000 stock-prediction-api

# Run with environment variables
docker run -p 8000:8000 -e NEWS_API_KEY=your_key stock-prediction-api
```

## 🚀 Production Deployment

### Using Gunicorn
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ⚠️ Important Disclaimers

### Investment Warning
- **Not Financial Advice**: This tool is for educational and informational purposes only
- **Market Volatility**: Stock markets are inherently unpredictable
- **Risk Assessment**: Always consult financial advisors before making investment decisions
- **Data Accuracy**: Predictions are based on historical data and may not reflect future performance

### Technical Limitations
- **Data Dependency**: Requires sufficient historical data (minimum 20 days)
- **Model Limitations**: LSTM models may not capture all market factors
- **External Factors**: Cannot account for sudden market events or news
- **Latency**: Predictions are based on previous day's closing prices

## 🔍 Troubleshooting

### Common Issues

#### 1. "Stock not found" Error
```bash
# Ensure ticker symbol is correct
# Try different exchange suffixes
curl http://localhost:8000/predict/RELIANCE  # NSE
curl http://localhost:8000/predict/RELIANCE.BO  # BSE
```

#### 2. Insufficient Data Error
```bash
# Check if stock has enough historical data
# Some newly listed stocks may not have sufficient data
```

#### 3. Model Training Issues
```bash
# Increase the data period
# Modify the get_stock_data function:
data, cmp = get_stock_data(ticker, period="2y")  # Use 2 years instead of 1
```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Optimization

### Model Performance
- **GPU Support**: Install PyTorch with CUDA for faster training
- **Batch Size**: Adjust batch size based on available memory
- **Feature Engineering**: Add more technical indicators

### API Performance
- **Caching**: Implement Redis caching for frequently requested stocks
- **Async Processing**: Use background tasks for model training
- **Load Balancing**: Deploy multiple instances behind a load balancer

## 🤝 Contributing

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance**: For providing free stock data API
- **FastAPI**: For the excellent web framework
- **PyTorch**: For the powerful neural network library
- **scikit-learn**: For data preprocessing utilities
- **Community**: For feedback and contributions

## 📞 Support

### Getting Help
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join community discussions
- **Documentation**: Check the FastAPI docs at `/docs` endpoint
- **Email**: Contact the maintainers for enterprise support

### FAQ

**Q: Which stocks are supported?**
A: All Indian stocks listed on NSE and BSE exchanges.

**Q: How accurate are the predictions?**
A: Accuracy varies based on market conditions and historical data quality. Use as one of many analysis tools.

**Q: Can I use this for live trading?**
A: This is for educational purposes. Do not use for automated trading without proper risk management.

**Q: How often should I retrain the model?**
A: Consider retraining weekly or when market conditions change significantly.

---

**⭐ Star this repository if you find it helpful!**

*Built with ❤️ for the Indian stock market community*
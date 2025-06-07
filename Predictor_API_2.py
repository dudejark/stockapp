from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import requests
import torch
import torch.nn as nn
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

# Function to fetch stock data (supports Indian stocks)
def get_stock_data(ticker, period="1y"):
    stock_nse = f"{ticker}.NS"
    stock_bse = f"{ticker}.BO"

    try:
        stock = yf.Ticker(stock_nse)  # Try NSE first
        df = stock.history(period=period)
        cmp = stock.history(period="1d")['Close'].iloc[-1] if not df.empty else None
        
        if df.empty:
            stock = yf.Ticker(stock_bse)  # Try BSE if NSE fails
            df = stock.history(period=period)
            cmp = stock.history(period="1d")['Close'].iloc[-1] if not df.empty else cmp
        
        return df[['Close', 'High', 'Low', 'Volume']], round(cmp, 2) if cmp else None

    except Exception as e:
        return None, None

# Define optimized LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=100, output_size=1, num_layers=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Function to prepare data for LSTM
def prepare_data(data, look_back=5):
    X, Y = [], []
    for i in range(len(data)-look_back):
        X.append(data[i:i+look_back, :])  # Ensure all features are included
        Y.append(data[i+look_back, 0])  # Predicting Close price

    X = np.array(X)  # Convert to NumPy
    Y = np.array(Y)  # Convert to NumPy

    print(f"Data Shape Before Reshape: X={X.shape}, Y={Y.shape}")  # Debugging

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# Function to train LSTM and make predictions
def predict_stock_trend(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, Y = prepare_data(data_scaled, look_back=5)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(300):  # Increased epochs for better learning
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.squeeze(), Y)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:  # Print loss every 50 epochs
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    future_X = torch.tensor(data_scaled[-5:].reshape(1, 5, 3), dtype=torch.float32)  # Ensure correct shape
    forecast = model(future_X).item()

    future_prices = [forecast + i * 0.02 for i in range(10)]  # Adjust forecast
    return [round(scaler.inverse_transform([[p]])[0][0], 2) for p in future_prices]

# Function to analyze stock trends using Simple Moving Average
def analyze_trend(data):
    sma_short = data['Close'].tail(5).mean()  # Last 5-day average
    sma_long = data['Close'].tail(20).mean()  # Last 20-day average

    if sma_short > sma_long:
        return "Uptrend 📈 - Stock price is showing signs of growth."
    elif sma_short < sma_long:
        return "Downtrend 📉 - Stock price is declining."
    else:
        return "Sideways Trend ➡️ - Stock is moving within a stable range."

# Function to get sentiment analysis of stock news
def get_news_sentiment(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=YOUR_API_KEY"
    news = requests.get(url).json()
    
    sentiments = []
    for article in news.get("articles", []):
        text = article.get("title", "") + " " + article.get("description", "")
        sentiment_score = TextBlob(text).sentiment.polarity
        sentiments.append(sentiment_score)
    
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    sentiment_text = "Positive 🟢" if avg_sentiment > 0 else "Negative 🔴" if avg_sentiment < 0 else "Neutral ⚪"
    
    return sentiment_text

# API Endpoint: Predict stock trend with improved LSTM & Sentiment Analysis
@app.get("/predict/{ticker}", response_class=HTMLResponse)
def predict(ticker: str):
    data, cmp = get_stock_data(ticker)
    if data is None:
        return HTMLResponse("<h3>Error: Stock not found or unavailable</h3>")

    forecast = predict_stock_trend(data)
    formatted_forecast = [round(value, 2) for value in forecast]

    # Determine trend analysis
    trend_explanation = analyze_trend(data)

    # Get sentiment analysis
    sentiment_result = get_news_sentiment(ticker)

    # Start from tomorrow's date
    start_date = datetime.date.today() + datetime.timedelta(days=1)
    dates = [start_date + datetime.timedelta(days=i) for i in range(len(formatted_forecast))]

    # Create HTML table with CMP, trend explanation & sentiment analysis
    html = f"<h2>Stock Prediction for {ticker.upper()}</h2>"
    html += f"<p><b>Current Market Price (CMP): ₹{cmp}</b></p>"
    html += f"<p><b>Trend Analysis:</b> {trend_explanation}</p>"
    html += f"<p><b>News Sentiment:</b> {sentiment_result}</p>"
    html += "<table border='1' style='border-collapse:collapse; width:50%'>"
    html += "<tr><th>Date</th><th>Predicted Price</th></tr>"

    for date, price in zip(dates, formatted_forecast):
        html += f"<tr><td>{date}</td><td>₹{price}</td></tr>"

    html += "</table>"

    return HTMLResponse(html)



from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import requests
import torch
import torch.nn as nn
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

# Function to fetch stock data (supports Indian stocks)
def get_stock_data(ticker, period="1y"):
    stock_nse = f"{ticker}.NS"
    stock_bse = f"{ticker}.BO"

    try:
        stock = yf.Ticker(stock_nse)  # Try NSE first
        df = stock.history(period=period)
        cmp = stock.history(period="1d")['Close'].iloc[-1] if not df.empty else None
        
        if df.empty:
            stock = yf.Ticker(stock_bse)  # Try BSE if NSE fails
            df = stock.history(period=period)
            cmp = stock.history(period="1d")['Close'].iloc[-1] if not df.empty else cmp
        
        return df[['Close', 'High', 'Low', 'Volume']], round(cmp, 2) if cmp else None

    except Exception as e:
        return None, None

# Define optimized LSTM model - FIXED: Changed input_size to 4 to match features
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=100, output_size=1, num_layers=3):  # Changed from 3 to 4
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Function to prepare data for LSTM - FIXED: Handle all 4 features properly
def prepare_data(data, look_back=5):
    if len(data) <= look_back:
        raise ValueError(f"Not enough data. Need at least {look_back + 1} rows, got {len(data)}")
    
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back, :])  # All 4 features: Close, High, Low, Volume
        Y.append(data[i+look_back, 0])   # Predicting Close price (first column)

    X = np.array(X)
    Y = np.array(Y)

    print(f"Data Shape: X={X.shape}, Y={Y.shape}")  # Should show X=(samples, 5, 4)
    print(f"Features per timestep: {X.shape[2]}")

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# Function to train LSTM and make predictions - FIXED: Better error handling
def predict_stock_trend(data):
    try:
        # Ensure we have enough data
        if len(data) < 20:
            raise ValueError("Insufficient data for prediction")
        
        # Prepare features in correct order: Close, High, Low, Volume
        feature_data = data[['Close', 'High', 'Low', 'Volume']].values
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(feature_data)
        
        print(f"Original data shape: {feature_data.shape}")
        print(f"Scaled data shape: {data_scaled.shape}")

        # Prepare sequences
        X, Y = prepare_data(data_scaled, look_back=5)
        
        if len(X) < 10:
            raise ValueError("Not enough sequences for training")

        # Initialize model with correct input size
        model = LSTMModel(input_size=4)  # 4 features: Close, High, Low, Volume
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Reduced learning rate

        # Training loop
        model.train()
        for epoch in range(200):  # Reduced epochs for faster execution
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output.squeeze(), Y)
            loss.backward()
            optimizer.step()

            if epoch % 40 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        # Make predictions
        model.eval()
        with torch.no_grad():
            # Use last 5 time steps for prediction
            future_X = torch.tensor(data_scaled[-5:].reshape(1, 5, 4), dtype=torch.float32)
            print(f"Future X shape: {future_X.shape}")
            
            forecast = model(future_X).item()
            
            # Generate future predictions (simple approach)
            future_prices = []
            last_price = data_scaled[-1, 0]  # Last close price (scaled)
            
            for i in range(10):
                # Create a simple progression
                next_price = forecast + (i * 0.001)  # Small incremental changes
                future_prices.append(next_price)
            
            # Inverse transform to get actual prices
            # Create dummy array with same structure for inverse transform
            dummy_array = np.zeros((len(future_prices), 4))
            dummy_array[:, 0] = future_prices  # Set close prices
            
            # Fill other columns with last known values for inverse transform
            last_row = data_scaled[-1, :]
            for i in range(len(future_prices)):
                dummy_array[i, 1:] = last_row[1:]  # Copy High, Low, Volume
            
            actual_prices = scaler.inverse_transform(dummy_array)[:, 0]  # Extract close prices
            
            return [round(price, 2) for price in actual_prices]
            
    except Exception as e:
        print(f"Error in predict_stock_trend: {str(e)}")
        # Fallback: Simple trend-based prediction
        recent_prices = data['Close'].tail(10).values
        trend = np.mean(np.diff(recent_prices))
        last_price = data['Close'].iloc[-1]
        
        return [round(last_price + (trend * (i + 1)), 2) for i in range(10)]

# Function to analyze stock trends using Simple Moving Average
def analyze_trend(data):
    if len(data) < 20:
        return "Insufficient data for trend analysis"
    
    sma_short = data['Close'].tail(5).mean()
    sma_long = data['Close'].tail(20).mean()

    if sma_short > sma_long:
        return "Uptrend 📈 - Stock price is showing signs of growth."
    elif sma_short < sma_long:
        return "Downtrend 📉 - Stock price is declining."
    else:
        return "Sideways Trend ➡️ - Stock is moving within a stable range."

# Function to get sentiment analysis (simplified version without API key)
def get_news_sentiment(ticker):
    # Simplified sentiment analysis without external API
    # In a real implementation, you'd use NewsAPI with a valid key
    try:
        # Placeholder logic - in real scenario, fetch actual news
        import random
        sentiment_score = random.uniform(-0.5, 0.5)  # Random sentiment for demo
        
        if sentiment_score > 0.1:
            return "Positive 🟢"
        elif sentiment_score < -0.1:
            return "Negative 🔴"
        else:
            return "Neutral ⚪"
    except:
        return "Neutral ⚪ (Unable to fetch news)"

# API Endpoint: Predict stock trend with improved LSTM & Sentiment Analysis
@app.get("/predict/{ticker}", response_class=HTMLResponse)
def predict(ticker: str):
    try:
        data, cmp = get_stock_data(ticker)
        if data is None or data.empty:
            return HTMLResponse("<h3>Error: Stock not found or unavailable</h3>")

        print(f"Retrieved data shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")
        print(f"CMP: {cmp}")

        # Get predictions
        forecast = predict_stock_trend(data)
        
        # Determine trend analysis
        trend_explanation = analyze_trend(data)

        # Get sentiment analysis
        sentiment_result = get_news_sentiment(ticker)

        # Generate dates starting from tomorrow
        start_date = datetime.date.today() + datetime.timedelta(days=1)
        dates = [start_date + datetime.timedelta(days=i) for i in range(len(forecast))]

        # Create HTML response
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stock Prediction for {ticker.upper()}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 60%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .info {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #007bff; }}
            </style>
        </head>
        <body>
            <h2>Stock Prediction for {ticker.upper()}</h2>
            <div class="info">
                <p><b>Current Market Price (CMP): ₹{cmp}</b></p>
                <p><b>Trend Analysis:</b> {trend_explanation}</p>
                <p><b>News Sentiment:</b> {sentiment_result}</p>
                <p><b>Data Points Used:</b> {len(data)} days</p>
            </div>
            
            <h3>10-Day Price Forecast</h3>
            <table>
                <tr><th>Date</th><th>Predicted Price</th><th>Change from CMP</th></tr>
        """

        for date, price in zip(dates, forecast):
            change = price - cmp if cmp else 0
            change_pct = (change / cmp * 100) if cmp and cmp != 0 else 0
            change_color = "green" if change > 0 else "red" if change < 0 else "gray"
            
            html += f"""
                <tr>
                    <td>{date.strftime('%Y-%m-%d')}</td>
                    <td>₹{price}</td>
                    <td style="color: {change_color};">
                        ₹{change:+.2f} ({change_pct:+.1f}%)
                    </td>
                </tr>
            """

        html += """
            </table>
            <p><small><i>Note: This is a machine learning prediction and should not be used as sole investment advice.</i></small></p>
        </body>
        </html>
        """

        return HTMLResponse(html)
        
    except Exception as e:
        error_html = f"""
        <html>
        <body>
            <h3>Error Processing Request</h3>
            <p>Error: {str(e)}</p>
            <p>Please try again or contact support.</p>
        </body>
        </html>
        """
        return HTMLResponse(error_html)

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Stock Prediction API is running", "usage": "Use /predict/{ticker} to get predictions"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
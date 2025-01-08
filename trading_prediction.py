from alpaca_trade_api.rest import REST, TimeFrame
import pandas as pd
import backtrader as bt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import talib
import sys

# Alpaca API credentials
API_KEY = 'AK1N4VMNGYEF42BFGXPQ'
SECRET_KEY = 'TI64nqrUXNqddFogFyl9Xbxr62qf80ro8GV72aFB'
BASE_URL = 'https://api.alpaca.markets'

# Step 1: Fetch Stock Data
def fetch_stock_data(stock_symbol, start_date, end_date):
    try:
        alpaca = REST(API_KEY, SECRET_KEY, BASE_URL)
        print(f"Fetching data for {stock_symbol} from {start_date} to {end_date}...")
        data = alpaca.get_bars(stock_symbol, TimeFrame.Day, start_date, end_date).df
        raw_file_name = f"{stock_symbol}_stock_data.csv"
        data.to_csv(raw_file_name, index=True)
        print(f"Raw data saved to {raw_file_name}")
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        sys.exit(1)

# Step 2: Load and Process the Dataset
def load_and_process_data(stock_symbol):
    file_name = f"{stock_symbol}_stock_data.csv"
    try:
        data = pd.read_csv(file_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_name}' was not found. Please ensure it exists.")

    print(f"Columns in the dataset: {data.columns}")

    # Rename the timestamp column to datetime
    if 'timestamp' in data.columns:
        data.rename(columns={'timestamp': 'datetime'}, inplace=True)

    if 'datetime' not in data.columns:
        raise ValueError("The 'datetime' column is missing in the dataset. Please check the data source.")

    # Convert datetime column to datetime64 format
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

    # Resample the data to ensure consistent intervals
    data = data.resample('D').ffill()

    # Add technical indicators using TA-Lib
    close = data['close'].values
    high = data['high'].values
    low = data['low'].values
    volume = data['volume'].values

    data['rsi'] = talib.RSI(close, timeperiod=14)
    data['macd'], data['macd_signal'], _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    data['bollinger_upper'], data['bollinger_middle'], data['bollinger_lower'] = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2)
    data['momentum'] = talib.MOM(close, timeperiod=10)
    data['atr'] = talib.ATR(high, low, close, timeperiod=14)
    data['stochastic_k'], data['stochastic_d'] = talib.STOCHF(high, low, close, fastk_period=14, fastd_period=3)
    data['cci'] = talib.CCI(high, low, close, timeperiod=20)
    data['adx'] = talib.ADX(high, low, close, timeperiod=14)

    # Create target column
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

    # Drop rows with NaN
    data = data.dropna()

    enhanced_file_name = f"{stock_symbol}_enhanced_stock_data.csv"
    data.to_csv(enhanced_file_name)
    print(f"Enhanced data saved to {enhanced_file_name}")
    return data

# Step 3: Train the Machine Learning Model
def train_model(data):
    features = ['rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower', 
                'momentum', 'atr', 'stochastic_k', 'stochastic_d', 'cci', 'adx']
    X = data[features]
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Optimize model with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight=class_weight_dict),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    model = grid_search.best_estimator_

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model

# Step 4: Define the Custom Data Feed
class CustomPandasData(bt.feeds.PandasData):
    lines = ('rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower', 
             'momentum', 'atr', 'stochastic_k', 'stochastic_d', 'cci', 'adx')
    params = {line: -1 for line in lines}
    params.update({'datetime': None, 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})

# Step 5: Define the Strategy
class MLStrategy(bt.Strategy):
    def __init__(self, model):
        self.model = model
        self.confidence_threshold = 0.75  # Confidence threshold for making decisions

    def next(self):
        # Extract features
        features = [self.data.__getattr__(line)[0] for line in [
            'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower', 
            'momentum', 'atr', 'stochastic_k', 'stochastic_d', 'cci', 'adx']]
        feature_df = pd.DataFrame([features], columns=[
            'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower', 
            'momentum', 'atr', 'stochastic_k', 'stochastic_d', 'cci', 'adx'])

        # Predict probabilities
        probabilities = self.model.predict_proba(feature_df)
        predicted_class = probabilities[0].argmax()
        confidence = probabilities[0][predicted_class]

        # Execute trades only if confidence >= confidence_threshold
        if confidence >= self.confidence_threshold:
            if predicted_class == 1 and not self.position:  # Buy signal
                buy_price = self.data.close[0]
                #print(f"BUY Signal at {self.data.datetime.datetime(0)}, Price: {buy_price}, Confidence: {confidence:.2f}")
                self.buy(size=10)
            elif predicted_class == 0 and self.position:  # Sell signal
                sell_price = self.data.close[0]
                #print(f"SELL Signal at {self.data.datetime.datetime(0)}, Price: {sell_price}, Confidence: {confidence:.2f}")
                self.sell(size=10)

# Step 6: Backtest with Backtrader
def run_backtest(data, model):
    bt_data = CustomPandasData(dataname=data)
    cerebro = bt.Cerebro()
    cerebro.adddata(bt_data)
    cerebro.addstrategy(MLStrategy, model=model)
    cerebro.broker.setcommission(commission=0.001)  # Transaction cost: 0.1%

    cerebro.run()
    cerebro.plot(style='line')

# Main Function
if __name__ == "__main__":
    from datetime import datetime, timedelta
    stock_symbol = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
    start_date = "2024-06-01"  # Fixed start date (can also be dynamic)
    start_date = sys.argv[2] if len(sys.argv) > 2 else "2024-01-01"
    end_date = sys.argv[3] if len(sys.argv) > 3 else datetime.now().strftime('%Y-%m-%d')

    # Dynamically set yesterday's date

    #yesterday = datetime.now() - timedelta(days=1)
    #end_date = yesterday.strftime('%Y-%m-%d')
    
    fetch_stock_data(stock_symbol, start_date, end_date)
    data = load_and_process_data(stock_symbol)
    model = train_model(data)
    run_backtest(data, model)

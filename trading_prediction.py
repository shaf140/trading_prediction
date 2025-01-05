from alpaca_trade_api.rest import REST, TimeFrame
import pandas as pd
import backtrader as bt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys

# Alpaca API credentials
API_KEY = 'apikey'
SECRET_KEY = 'secret'
BASE_URL = 'https://api.alpaca.markets'

# Step 1: Fetch Stock Data
def fetch_stock_data(stock_symbol, start_date, end_date):
    alpaca = REST(API_KEY, SECRET_KEY, BASE_URL)
    print(f"Fetching data for {stock_symbol} from {start_date} to {end_date}...")
    data = alpaca.get_bars(stock_symbol, TimeFrame.Day, start_date, end_date).df
    raw_file_name = f"{stock_symbol}_stock_data.csv"
    data.to_csv(raw_file_name, index=True)
    print(f"Raw data saved to {raw_file_name}")
    return data

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
    
    # Ensure the datetime column exists
    if 'datetime' not in data.columns:
        raise ValueError("The 'datetime' column is missing in the dataset. Please check the data source.")

    # Convert datetime column to datetime64 format
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

    # Resample the data to ensure consistent intervals
    data = data.resample('D').ffill()  # Daily frequency with forward-fill for missing values

    # Add technical indicators
    # Moving Averages
    data['moving_avg_10'] = data['close'].rolling(window=10).mean()
    data['moving_avg_50'] = data['close'].rolling(window=50).mean()

    # MACD and Signal Line
    data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = data['ema_12'] - data['ema_26']
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

    # RSI
    data['rsi'] = 100 - (100 / (1 + (data['close'].diff().clip(lower=0).sum() / abs(data['close'].diff().clip(upper=0).sum()))))

    # Bollinger Bands
    data['bollinger_upper'] = data['moving_avg_10'] + 2 * data['close'].rolling(window=10).std()
    data['bollinger_lower'] = data['moving_avg_10'] - 2 * data['close'].rolling(window=10).std()

    # ATR
    data['atr'] = (data['high'] - data['low']).rolling(window=14).mean()

    # Stochastic Oscillator
    data['stochastic_k'] = ((data['close'] - data['low'].rolling(14).min()) /
                            (data['high'].rolling(14).max() - data['low'].rolling(14).min())) * 100

    # OBV
    data['obv'] = (data['volume'] * ((data['close'].diff() > 0).astype(int) -
                                     (data['close'].diff() < 0).astype(int))).cumsum()

    # VWAP
    data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()

    # Williams %R
    data['williams_r'] = ((data['high'].rolling(14).max() - data['close']) /
                         (data['high'].rolling(14).max() - data['low'].rolling(14).min())) * -100

    # Chaikin Money Flow (CMF)
    data['cmf'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / \
                  (data['high'] - data['low']) * data['volume']

    # Advanced Indicators
    data['alma'] = (data['close'] * 0.85) + (data['close'].rolling(10).mean() * 0.15)
    data['adx'] = ((data['high'] - data['low']).rolling(14).mean())
    data['parabolic_sar'] = data['close'].rolling(window=2).mean()
    data['momentum'] = data['close'] - data['close'].shift(10)
    data['cci'] = (data['close'] - data['close'].rolling(20).mean()) / (0.015 * data['close'].rolling(20).std())

    # Create target column
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

    # Drop rows with NaN
    data = data.dropna(subset=[
        'moving_avg_10', 'moving_avg_50', 'macd', 'macd_signal', 'rsi',
        'bollinger_upper', 'bollinger_lower', 'atr', 'stochastic_k', 'obv',
        'vwap', 'williams_r', 'cmf', 'alma', 'adx', 'parabolic_sar', 'momentum', 'cci', 'target'
    ])

    enhanced_file_name = f"{stock_symbol}_enhanced_stock_data.csv"
    data.to_csv(enhanced_file_name)
    print(f"Enhanced data saved to {enhanced_file_name}")
    return data

# Step 3: Train the Machine Learning Model
def train_model(data):
    X = data[['moving_avg_10', 'moving_avg_50', 'macd', 'macd_signal', 'rsi',
              'bollinger_upper', 'bollinger_lower', 'atr', 'stochastic_k', 'obv',
              'vwap', 'williams_r', 'cmf', 'alma', 'adx', 'parabolic_sar', 'momentum', 'cci']]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model

# Step 4: Define the Custom Data Feed
class CustomPandasData(bt.feeds.PandasData):
    lines = ('moving_avg_10', 'moving_avg_50', 'macd', 'macd_signal', 'rsi',
             'bollinger_upper', 'bollinger_lower', 'atr', 'stochastic_k', 'obv',
             'vwap', 'williams_r', 'cmf', 'alma', 'adx', 'parabolic_sar', 'momentum', 'cci')
    params = {line: -1 for line in lines}
    params.update({
        'datetime': None,
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })

# Step 5: Define the Strategy
class MLStrategy(bt.Strategy):
    def __init__(self, model):
        self.model = model

    def next(self):
        features = [self.data.__getattr__(line)[0] for line in [
            'moving_avg_10', 'moving_avg_50', 'macd', 'macd_signal', 'rsi',
            'bollinger_upper', 'bollinger_lower', 'atr', 'stochastic_k', 'obv',
            'vwap', 'williams_r', 'cmf', 'alma', 'adx', 'parabolic_sar', 'momentum', 'cci']]
        feature_df = pd.DataFrame([features], columns=[
            'moving_avg_10', 'moving_avg_50', 'macd', 'macd_signal', 'rsi',
            'bollinger_upper', 'bollinger_lower', 'atr', 'stochastic_k', 'obv',
            'vwap', 'williams_r', 'cmf', 'alma', 'adx', 'parabolic_sar', 'momentum', 'cci'])
        prediction = self.model.predict(feature_df)
        if prediction[0] == 1 and not self.position:
            self.buy(size=10)
        elif prediction[0] == 0 and self.position:
            self.sell(size=10)

# Step 6: Backtest with Backtrader
def run_backtest(data, model):
    bt_data = CustomPandasData(dataname=data)
    cerebro = bt.Cerebro()
    cerebro.adddata(bt_data)
    cerebro.addstrategy(MLStrategy, model=model)
    cerebro.run()
    cerebro.plot()

# Main Function
if __name__ == "__main__":
    stock_symbol = sys.argv[1] if len(sys.argv) > 1 else "DJI"
    start_date = sys.argv[2] if len(sys.argv) > 2 else "2024-01-01"
    end_date = sys.argv[3] if len(sys.argv) > 3 else "2025-01-01"
    fetch_stock_data(stock_symbol, start_date, end_date)
    data = load_and_process_data(stock_symbol)
    model = train_model(data)
    run_backtest(data, model)
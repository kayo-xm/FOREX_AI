import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, LSTM, Dense, Dropout, 
                                   Flatten, Concatenate)
from transformers import TFBertModel
import talib
import logging

# --- Structure & Price Action Feature Extraction ---
def detect_swings(df, lookback=5):
    """
    Detect swing highs and lows.
    Returns two Series: swing_highs, swing_lows (1 if swing, else 0)
    """
    swing_highs = (df['high'] == df['high'].rolling(window=lookback*2+1, center=True).max()).astype(int)
    swing_lows = (df['low'] == df['low'].rolling(window=lookback*2+1, center=True).min()).astype(int)
    return swing_highs, swing_lows

def label_structure(df, swing_highs, swing_lows):
    """
    Label the structure as HH, HL, LH, LL
    """
    structure = []
    last_high = last_low = None
    for idx in range(len(df)):
        label = ''
        if swing_highs.iloc[idx]:
            if last_high is not None and df['high'].iloc[idx] > last_high:
                label = 'HH'
            elif last_high is not None:
                label = 'LH'
            last_high = df['high'].iloc[idx]
        elif swing_lows.iloc[idx]:
            if last_low is not None and df['low'].iloc[idx] > last_low:
                label = 'HL'
            elif last_low is not None:
                label = 'LL'
            last_low = df['low'].iloc[idx]
        structure.append(label)
    return pd.Series(structure, index=df.index)

def price_action_features(df):
    """
    Extract basic price action features (engulfing, pin bar, inside bar)
    Returns a DataFrame with binary columns for each pattern.
    """
    features = pd.DataFrame(index=df.index)
    # Bullish engulfing
    features['bullish_engulfing'] = (
        (df['close'] > df['open']) &
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['close'] > df['open'].shift(1)) &
        (df['open'] < df['close'].shift(1))
    ).astype(int)
    # Bearish engulfing
    features['bearish_engulfing'] = (
        (df['close'] < df['open']) &
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['open'] > df['close'].shift(1)) &
        (df['close'] < df['open'].shift(1))
    ).astype(int)
    # Pin bar (bullish)
    body = abs(df['close'] - df['open'])
    candle_range = df['high'] - df['low']
    features['bullish_pinbar'] = ((df['low'] - df[['open', 'close']].min(axis=1) > body * 2) &
                                  (body / candle_range < 0.3)).astype(int)
    # Inside bar
    features['inside_bar'] = (
        (df['high'] < df['high'].shift(1)) &
        (df['low'] > df['low'].shift(1))
    ).astype(int)
    return features

class ForexAnalysisModel:
    def __init__(self):
        # Multi-modal input
        self.chart_input = Input(shape=(100, 100, 3), name='chart_input')
        self.time_series_input = Input(shape=(60, 5), name='time_series_input')
        self.text_input = Input(shape=(128,), name='text_input', dtype='int32')
        # Build ensemble model
        self.model = self.build_ensemble_model()
        
    def build_cnn_branch(self):
        x = Conv2D(32, (3,3), activation='relu')(self.chart_input)
        x = Conv2D(64, (3,3), activation='relu')(x)
        x = Conv2D(128, (3,3), activation='relu')(x)
        x = Flatten()(x)
        return Dense(64, activation='relu')(x)
    
    def build_lstm_branch(self):
        x = LSTM(64, return_sequences=True)(self.time_series_input)
        x = LSTM(32)(x)
        return Dense(16, activation='relu')(x)
    
    def build_bert_branch(self):
        try:
            bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
        except Exception as e:
            logging.error(f"BERT model could not be loaded: {e}")
            raise
        text_embedding = bert_layer(self.text_input)[0][:,0,:]
        return Dense(32, activation='relu')(text_embedding)
    
    def build_ensemble_model(self):
        cnn_branch = self.build_cnn_branch()
        lstm_branch = self.build_lstm_branch()
        bert_branch = self.build_bert_branch()
        merged = Concatenate()([cnn_branch, lstm_branch, bert_branch])
        x = Dense(128, activation='relu')(merged)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        direction = Dense(3, activation='softmax', name='direction')(x)
        confidence = Dense(1, activation='sigmoid', name='confidence')(x)
        risk = Dense(1, activation='relu', name='risk_score')(x)
        return Model(
            inputs=[self.chart_input, self.time_series_input, self.text_input],
            outputs=[direction, confidence, risk]
        )
    
    def preprocess_market_data(self, df):
        """
        Create features from raw market data, including structure and price action.
        """
        # Ensure columns are lower case for feature extraction
        df = df.rename(columns=str.lower)
        features = pd.DataFrame()
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['close'] = df['close']
        features['volume'] = df['volume']

        # --- Structure features ---
        swing_highs, swing_lows = detect_swings(df)
        structure_labels = label_structure(df, swing_highs, swing_lows)
        features['is_swing_high'] = swing_highs
        features['is_swing_low'] = swing_lows
        features['structure_label'] = structure_labels.replace({'HH':2, 'HL':1, 'LH':-1, 'LL':-2, '':0})

        # --- Price action features ---
        pa = price_action_features(df)
        features = pd.concat([features, pa], axis=1)

        # --- Technical indicators ---
        try:
            features['rsi'] = talib.RSI(df['close'], timeperiod=14)
            features['macd'], _, _ = talib.MACD(df['close'])
            features['bollinger_upper'], features['bollinger_middle'], features['bollinger_lower'] = talib.BBANDS(df['close'])
        except Exception as e:
            logging.warning(f"TA-Lib computation error: {e}")

        return features

    def train(self, X_chart, X_series, X_text, y_direction, y_confidence):
        self.model.compile(
            optimizer='adam',
            loss={
                'direction': 'categorical_crossentropy',
                'confidence': 'mse',
                'risk_score': 'mse'
            },
            metrics=['accuracy']
        )
        self.model.fit(
            {'chart_input': X_chart, 'time_series_input': X_series, 'text_input': X_text},
            {'direction': y_direction, 'confidence': y_confidence, 'risk_score': y_confidence},
            epochs=50,
            batch_size=32,
            validation_split=0.2
        )
from fastapi import FastAPI, UploadFile, File, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import pandas as pd
import yfinance as yf
import talib
from tensorflow.keras.models import load_model
from datetime import datetime
import cv2
import asyncio
import json
import os
from decimal import Decimal

from ml.pipeline import ForexAnalysisModel
from trading.engine import TradingEngine, TradeSignal

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve built frontend (React) ---
# Assumes your frontend is built and output is in ./frontend/build (relative to this file)
if os.path.isdir("frontend/build"):
    app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")

# --- Healthcheck endpoint (for Docker, monitoring, etc.) ---
@app.get("/api/health")
def health():
    return {"status": "ok"}

# --- Load ML Models ---
models = {
    'pattern_cnn': load_model('models/pattern_cnn.h5'),
    'trend_lstm': load_model('models/trend_lstm.h5'),
    'sentiment_bert': load_model('models/sentiment_bert.h5')
}
ml_pipeline = ForexAnalysisModel()
trading_engine = None  # Will be initialized via /api/start-trading

# --- API Models ---
class StrategyRequest(BaseModel):
    api_key: str
    api_secret: str
    pairs: List[str]
    rules: Dict
    interval: int  # in seconds

class SignalRequest(BaseModel):
    direction: str  # 'BUY' or 'SELL'
    pair: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    size: float

# --- Endpoints ---

@app.post("/api/analyze")
async def analyze_market(
    pair: str, 
    timeframe: str, 
    patterns: List[UploadFile] = File(...)
):
    market_data = await get_multi_timeframe_data(pair, timeframe)
    tech_indicators = calculate_technical_indicators(market_data['primary'])
    pattern_results = await analyze_chart_patterns(patterns)
    sentiment = await analyze_market_sentiment(pair)
    signal = generate_trading_signal(
        market_data, tech_indicators, pattern_results, sentiment
    )
    return {
        "signal": signal['direction'],
        "confidence": signal['confidence'],
        "entry_points": signal.get('entry_points', []),
        "exit_points": signal.get('exit_points', []),
        "risk_reward": signal.get('risk_reward', 1.0),
        "market_data": {k: v.to_dict() if hasattr(v, 'to_dict') else v for k,v in market_data.items()},
        "technical_indicators": tech_indicators,
        "pattern_analysis": pattern_results,
        "sentiment_analysis": sentiment
    }

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            pair = json.loads(data)['pair']
            while True:
                live_data = get_live_data(pair)
                await websocket.send_json(live_data)
                await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.post("/api/start-trading")
async def start_trading(strategy: StrategyRequest, background_tasks: BackgroundTasks):
    global trading_engine
    trading_engine = TradingEngine(strategy.api_key, strategy.api_secret)
    background_tasks.add_task(trading_engine.execute_strategy, strategy.dict())
    return {"status": "Trading engine started"}

@app.post("/api/execute-signal")
async def execute_signal(signal: SignalRequest):
    if not trading_engine:
        return {"error": "Trading engine not started"}
    signal_obj = TradeSignal(
        direction=signal.direction,
        pair=signal.pair,
        entry_price=Decimal(str(signal.entry_price)),
        stop_loss=Decimal(str(signal.stop_loss)),
        take_profit=Decimal(str(signal.take_profit)),
        confidence=signal.confidence,
        size=Decimal(str(signal.size)),
        timestamp=int(datetime.utcnow().timestamp())
    )
    asyncio.create_task(trading_engine.place_trade(signal_obj))
    return {"status": "Signal sent to trading engine"}

# --- Helper functions ---

async def get_multi_timeframe_data(pair: str, primary_tf: str):
    """Fetch data for multiple timeframes for confluence analysis."""
    timeframes = {
        'higher': get_higher_timeframe(primary_tf),
        'primary': primary_tf,
        'lower': get_lower_timeframe(primary_tf)
    }
    data = {}
    for key, tf in timeframes.items():
        df = yf.download(f"{pair}=X", period="60d", interval=tf)
        data[key] = preprocess_market_data(df)
    return data

def calculate_technical_indicators(df):
    """Calculate advanced technical indicators."""
    indicators = {
        'rsi': talib.RSI(df['close'], timeperiod=14).tolist() if 'close' in df else [],
        'macd': talib.MACD(df['close'])[0].tolist() if 'close' in df else [],
        'atr': talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).tolist() if all(x in df for x in ('high','low','close')) else [],
        'ichimoku': calculate_ichimoku(df),
        'fibonacci': calculate_fibonacci_levels(df),
        'volume_profile': calculate_volume_profile(df)
    }
    return indicators

async def analyze_chart_patterns(files):
    """Process uploaded chart patterns with CNN"""
    pattern_results = []
    for file in files:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        img = preprocess_image(img)
        prediction = models['pattern_cnn'].predict(np.expand_dims(img, axis=0))
        pattern_results.append({
            'pattern': decode_prediction(prediction),
            'confidence': float(np.max(prediction))
        })
    return pattern_results

async def analyze_market_sentiment(pair):
    """Perform sentiment analysis using your BERT model (implement as needed)."""
    # Placeholder: Replace with real news/social media sentiment code
    return {"sentiment": "neutral", "confidence": 0.5}

def generate_trading_signal(market_data, tech_indicators, pattern_results, sentiment):
    """Combine all signals to produce a trading recommendation."""
    # Placeholder: Replace with your strategy logic or use ml_pipeline.model.predict
    return {
        "direction": "HOLD",
        "confidence": 0.5,
        "entry_points": [],
        "exit_points": [],
        "risk_reward": 1.0
    }

def get_live_data(pair):
    """Fetch live market data (implement with real API or yfinance)."""
    df = yf.download(f"{pair}=X", period="1d", interval="1m")
    if not df.empty:
        last = df.iloc[-1]
        return {
            "timestamp": int(last.name.timestamp()),
            "open": float(last['Open']),
            "high": float(last['High']),
            "low": float(last['Low']),
            "close": float(last['Close']),
            "volume": float(last['Volume'])
        }
    return {}

def preprocess_market_data(df):
    df = df.rename(columns=str.lower)
    df = df.fillna(method='ffill')
    return df

def get_higher_timeframe(tf):
    mapping = {"1m":"5m","5m":"15m","15m":"1h","1h":"4h","4h":"1d","1d":"1wk"}
    return mapping.get(tf, tf)

def get_lower_timeframe(tf):
    mapping = {"1wk":"1d","1d":"4h","4h":"1h","1h":"15m","15m":"5m","5m":"1m"}
    return mapping.get(tf, tf)

def calculate_ichimoku(df): return {}
def calculate_fibonacci_levels(df): return {}
def calculate_volume_profile(df): return {}

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img

def decode_prediction(prediction):
    pattern_labels = ["Bullish", "Bearish", "Neutral"]
    idx = int(np.argmax(prediction))
    return pattern_labels[idx] if idx < len(pattern_labels) else "Unknown"
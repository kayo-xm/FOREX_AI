# Forex AI Pattern Analyzer

![Forex AI Dashboard](https://i.imgur.com/JkQ2RZT.png)

A comprehensive AI-powered platform for Forex market pattern recognition, analysis, and automated trading strategies.

## Features

- **Real-time Market Analysis**: Live price data with AI pattern detection
- **Multi-Timeframe Analysis**: Simultaneous analysis across different timeframes
- **Incremental Learning**: Continuously improves pattern recognition
- **Advanced Visualization**: TradingView-like charts with pattern annotations
- **Risk Management Tools**: Position sizing calculator and risk assessment

## Technologies

- **Frontend**: React 18, TypeScript, Redux Toolkit, Lightweight Charts
- **Backend**: FastAPI, Python, TensorFlow, TA-Lib
- **Database**: MongoDB, Redis
- **DevOps**: Docker, GitHub Actions, NGINX

## Getting Started

### Prerequisites

- Docker 20.10+
- Node.js 18+
- Python 3.10+
- NVIDIA GPU (recommended for training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kayo-xm/FOREX_AI.git
   cd FOREX_AI
   ```

2. **Copy the environment example and edit as needed:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and secrets
   ```

3. **Place model weights:**
   - Upload your trained model files (`pattern_cnn.h5`, `trend_lstm.h5`, `sentiment_bert.h5`) into the `models/` directory inside the repo root.

4. **Build and start all services:**
   ```bash
   docker compose up --build -d
   ```

5. **Build and deploy the frontend:**
   ```bash
   cd frontend
   npm install
   npm run build
   ```
   - For production: Serve the `build/` folder with Netlify/Vercel/NGINX, or set up FastAPI to serve static files.

6. **Access the app:**
   - Backend API: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Frontend: [http://localhost:3000/](http://localhost:3000/) (or your chosen host)

### Development

- Use the provided `docker-compose.yml` for local development.
- To run backend or frontend individually, see respective directories for details.

### Security

- **Never commit real secrets.**
- Change all default passwords before deploying to production.
- Use HTTPS and firewall your server.

## Troubleshooting

- **Healthcheck fails:** Make sure `/api/health` endpoint exists and returns `{"status": "ok"}`.
- **Model loading errors:** Ensure `.h5` files are present in `/models` directory.
- **Frontend not loading data:** Check `REACT_APP_API_URL` and backend CORS settings.

## License

MIT
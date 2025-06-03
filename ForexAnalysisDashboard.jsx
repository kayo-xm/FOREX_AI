import React, { useState, useEffect, useRef } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import { PairSelector } from './components/PairSelector';
import { TimeframeSelector } from './components/TimeframeSelector';
import { PatternUploader } from './components/PatternUploader';
import { TechnicalAnalysis } from './components/TechnicalAnalysis';
import { PatternRecognition } from './components/PatternRecognition';
import { SentimentAnalysis } from './components/SentimentAnalysis';
import { TradingSignals } from './components/TradingSignals';

const ForexAnalysisDashboard = () => {
  const [pair, setPair] = useState('EURUSD');
  const [timeframe, setTimeframe] = useState('1h');
  const [analysis, setAnalysis] = useState(null);
  const [liveData, setLiveData] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const ws = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket(`ws://localhost:8000/ws/live`);
    ws.current.onopen = () => {
      ws.current.send(JSON.stringify({ pair }));
    };
    ws.current.onmessage = (e) => {
      const data = JSON.parse(e.data);
      setLiveData(prev => [...prev.slice(-100), data]);
    };
    return () => ws.current.close();
  }, [pair]);

  const handleAnalysis = async (uploadedFiles) => {
    setIsAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append('pair', pair);
      formData.append('timeframe', timeframe);
      uploadedFiles.forEach(file => formData.append('patterns', file));
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        body: formData
      });
      const result = await response.json();
      setAnalysis(result);
    } catch (error) {
      console.error('Analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="forex-dashboard">
      <div className="control-panel">
        <PairSelector value={pair} onChange={setPair} />
        <TimeframeSelector value={timeframe} onChange={setTimeframe} />
        <PatternUploader onAnalyze={handleAnalysis} />
      </div>
      <div className="main-content">
        <div className="chart-section">
          <TVChartContainer 
            pair={pair} 
            timeframe={timeframe}
            analysis={analysis}
            liveData={liveData}
          />
        </div>
        <div className="analysis-section">
          <TechnicalAnalysis data={analysis?.technical_indicators} />
          <PatternRecognition patterns={analysis?.pattern_analysis} />
          <SentimentAnalysis sentiment={analysis?.sentiment_analysis} />
          <TradingSignals signal={analysis?.signal} />
        </div>
      </div>
    </div>
  );
};

export const TVChartContainer = ({ pair, timeframe, analysis, liveData }) => {
  const chartContainerRef = useRef();
  const seriesRef = useRef();

  useEffect(() => {
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1e293b' },
        textColor: '#f8fafc',
      },
      grid: {
        vertLines: { color: '#334155' },
        horzLines: { color: '#334155' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 500,
    });
    const candleSeries = chart.addCandlestickSeries();
    seriesRef.current = candleSeries;
    return () => chart.remove();
  }, []);

  useEffect(() => {
    if (seriesRef.current && liveData.length) {
      const formattedData = liveData.map(item => ({
        time: item.timestamp,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close
      }));
      seriesRef.current.setData(formattedData);
      if (analysis) {
        const markers = (analysis.entry_points || []).map(point => ({
          time: point.time,
          position: 'belowBar',
          color: '#00ff00',
          shape: 'arrowUp',
          text: 'Entry'
        })).concat((analysis.exit_points || []).map(point => ({
          time: point.time,
          position: 'aboveBar',
          color: '#ff0000',
          shape: 'arrowDown',
          text: 'Exit'
        })));
        seriesRef.current.setMarkers(markers);
      }
    }
  }, [liveData, analysis]);

  return <div ref={chartContainerRef} className="advanced-chart" />;
};

export default ForexAnalysisDashboard;
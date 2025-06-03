import React from "react";
export const SentimentAnalysis = ({ sentiment }) => (
  <div>
    <h3>Sentiment</h3>
    {sentiment
      ? <div>{sentiment.sentiment} ({(sentiment.confidence*100).toFixed(1)}%)</div>
      : <div>No sentiment data</div>
    }
  </div>
);
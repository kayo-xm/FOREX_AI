import React from "react";
const TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"];
export const TimeframeSelector = ({ value, onChange }) => (
  <select value={value} onChange={e => onChange(e.target.value)}>
    {TIMEFRAMES.map(tf => (
      <option key={tf} value={tf}>{tf}</option>
    ))}
  </select>
);
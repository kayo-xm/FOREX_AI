import React, { useRef } from "react";
export const PatternUploader = ({ onAnalyze }) => {
  const inputRef = useRef();
  const handleFileChange = (e) => {
    onAnalyze(Array.from(e.target.files));
  };
  return (
    <input
      type="file"
      multiple
      accept="image/*"
      ref={inputRef}
      onChange={handleFileChange}
    />
  );
};
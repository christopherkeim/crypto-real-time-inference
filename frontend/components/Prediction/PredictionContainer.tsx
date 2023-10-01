"use client";

import { useState } from "react";
import { PredictionForm, Prediction } from "./PredictionForm";
import { PredictionResult } from "./PredictionResult";

export function PredictionContainer() {
  const [prediction, setPrediction] = useState<Prediction | null>(null);

  return (
    <>
      <PredictionForm setPrediction={setPrediction} />
      <PredictionResult prediction={prediction} />
    </>
  );
}

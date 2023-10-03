import { Prediction } from "./PredictionForm";

type PredictionResultProps = {
  prediction: Prediction | null;
};

export function PredictionResult({ prediction }: PredictionResultProps) {
  return (
    <div>
      {/* TODO: add animation to load content */}
      {prediction && (
        <p className="text-xl text-gray-600 dark:text-gray-400">
          {`The ${prediction.coin} price prediction is $${
            prediction.prediction.amount
          } at ${prediction.prediction.time.toTimeString()}. This prediction used the ${
            prediction.model
          } model.`}
        </p>
      )}
    </div>
  );
}

"use client";

import { Dispatch, SetStateAction, useState } from "react";

type PredictionFormProps = {
  setPrediction: Dispatch<SetStateAction<Prediction | null>>;
};

export type Prediction = {
  coin: string;
  time: string;
  model: string;
  prediction: { amount: number; time: Date };
};

export function PredictionForm({ setPrediction }: PredictionFormProps) {
  const [coin, setCoin] = useState<string>("BTC");
  const [time, setTime] = useState<string>("CURRENT");
  const [model, setModel] = useState<string>("BTC-1HR-CNN");

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    try {
      // Fire off fetch to predict endpoint
      // TODO: Change this to the real endpoint
      const response = await fetch("/mock", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          coin: coin,
          time: time,
          model: model,
        }),
      });

      if (!response.ok)
        throw new Error(
          "An error occurred while performing api fetch: " +
            response.statusText,
        );

      const json = await response.json();

      if (!json.prediction)
        throw new Error("An error occurred while parsing response: " + json);

      // If all goes well, set the prediction state, which will trigger a re-render in the prediction result component
      setPrediction({
        coin: json.coin,
        time: json.time,
        model: json.model,
        prediction: {
          amount: json.prediction.amount,
          time: new Date(json.prediction.time),
        },
      });
    } catch (error) {
      console.log(error);
    }
  }

  return (
    <form
      className="mt-8 flex w-full flex-col gap-1 md:flex-row md:gap-4"
      data-aos="fade-down"
      data-aos-delay="300"
      onSubmit={handleSubmit}
    >
      <div className="flex flex-col">
        <label
          htmlFor="coin"
          className="mb-2 block text-sm font-medium text-gray-900 dark:text-white"
        >
          Coin
        </label>
        <select
          className="form-input mb-2 mr-10 w-full py-2 md:mr-4"
          id="coin"
          aria-label="Select Coin"
          onChange={(event) => setCoin(event.target.value)}
        >
          <option value="BTC">BTC</option>
          {/* <option value="ETC">ETC</option> */}
        </select>
      </div>

      <div className="flex grow flex-col">
        <label
          htmlFor="time"
          className="mb-2 block text-sm font-medium text-gray-900 dark:text-white"
        >
          Time From
        </label>
        <select
          className="form-input mb-2 mr-4 w-full py-2"
          id="time"
          aria-label="Coin"
          onChange={(event) => setTime(event.target.value)}
        >
          <option value="CURRENT">Current Time</option>
          <option value="LESS-1HR">Current Time - 1 Hour</option>
          <option value="LESS-2HR">Current Time - 2 Hour</option>
          <option value="LESS-3HR">Current Time - 3 Hour</option>
          <option value="LESS-4HR">Current Time - 4 Hour</option>
          <option value="LESS-5HR">Current Time - 5 Hour</option>
          <option value="LESS-6HR">Current Time - 6 Hour</option>
          <option value="LESS-7HR">Current Time - 7 Hour</option>
          <option value="LESS-8HR">Current Time - 8 Hour</option>
          <option value="LESS-9HR">Current Time - 9 Hour</option>
        </select>
      </div>

      <div className="flex grow flex-col">
        <label
          htmlFor="model"
          className="mb-2 block text-sm font-medium text-gray-900 dark:text-white"
        >
          Model
        </label>
        <select
          className="form-input mb-2 mr-4 w-full py-2"
          id="model"
          aria-label="Select Model"
          onChange={(event) => setModel(event.target.value)}
        >
          <option value="BTC-1HR-CNN">BTC +1 Hour (CNN)</option>
          <option value="BTC-1HR-LASSO">BTC +1 Hour (LASSO)</option>
          {/* <option value="ETC-1HR-CNN">ETC +1 Hour (CNN)</option> */}
          {/* <option value="ETC-1HR-LASSO">ETC +1 Hour (LASSO)</option> */}
        </select>
      </div>

      <button
        type="submit"
        className="btn mb-2 mt-auto h-10 bg-blue-500 text-white hover:bg-blue-400"
      >
        Predict
      </button>
    </form>
  );
}

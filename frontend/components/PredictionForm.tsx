"use client";

export function PredictionForm() {
  return (
    <form
      className="mt-8 w-full flex flex-col md:flex-row gap-1 md:gap-4"
      data-aos="fade-down"
      data-aos-delay="300"
      onSubmit={(event) => {
        event.preventDefault();
        console.log(event);
        // fetch from server
      }}
    >
      <div className="flex flex-col">
        <label
          htmlFor="coin"
          className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
        >
          Coin
        </label>
        <select
          className="form-input w-full mb-2 mr-10 md:mr-4 py-2"
          id="coin"
          aria-label="Select Coin"
        >
          <option value="BTC">BTC</option>
          {/* <option value="ETC">ETC</option> */}
        </select>
      </div>

      <div className="flex flex-col grow">
        <label
          htmlFor="time"
          className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
        >
          Time From
        </label>
        <select
          className="form-input w-full mb-2 mr-4 py-2"
          id="time"
          aria-label="Coin"
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

      <div className="flex flex-col grow">
        <label
          htmlFor="model"
          className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
        >
          Model
        </label>
        <select
          className="form-input w-full mb-2 mr-4 py-2"
          id="model"
          aria-label="Select Model"
        >
          <option value="BTC-1HR-CNN">BTC +1 Hour (CNN)</option>
          <option value="BTC-1HR-LASSO">BTC +1 Hour (LASSO)</option>
          {/* <option value="ETC-1HR-CNN">ETC +1 Hour (CNN)</option> */}
          {/* <option value="ETC-1HR-LASSO">ETC +1 Hour (LASSO)</option> */}
        </select>
      </div>

      <button
        type="submit"
        className="btn text-white bg-blue-500 hover:bg-blue-400 h-10 mt-auto mb-2"
      >
        Predict
      </button>
    </form>
  );
}

import React from "react";

type ProgressBarProps = {
  className?: string;
  value: "beginner" | "intermediate" | "advanced";
  valueMin?: number;
  valueMax?: number;
};

enum ProgressBarValue {
  beginner = 25,
  intermediate = 60,
  advanced = 90,
}

export function ProgressBar({
  className,
  value,
  valueMin = 0,
  valueMax = 100,
}: ProgressBarProps) {
  return (
    <div
      className={`relative h-1.5 bg-slate-200 before:absolute before:inset-0 before:w-[${ProgressBarValue[value]}%] before:bg-sky-500 dark:bg-slate-700 ${className}`}
      role="progressbar"
      aria-valuenow={ProgressBarValue[value]}
      aria-valuemin={valueMin}
      aria-valuemax={valueMax}
    ></div>
  );
}

export default ProgressBar;

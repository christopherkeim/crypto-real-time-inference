import React from "react";

type ProgressBarProps = {
  className?: string;
  value: number;
  valueMin?: number;
  valueMax?: number;
};

export function ProgressBar({
  className,
  value,
  valueMin = 0,
  valueMax = 100,
}: ProgressBarProps) {
  return (
    <div
      className={`relative h-1.5 bg-slate-200 before:absolute before:inset-0 before:w-[${value}%] before:bg-sky-500 dark:bg-slate-700 ${className}`}
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={valueMin}
      aria-valuemax={valueMax}
    ></div>
  );
}

export default ProgressBar;

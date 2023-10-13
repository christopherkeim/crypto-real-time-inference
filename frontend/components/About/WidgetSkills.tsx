import { ProgressBar } from "@/components/About/ProgressBar";

export function WidgetSkills() {
  return (
    <section className="rounded-lg border border-slate-200 p-5 dark:border-slate-800 dark:bg-gradient-to-t dark:from-slate-800 dark:to-slate-800/30">
      <div className="mb-3 font-[650]">Technical Skills</div>
      <ul className="space-y-3">
        <li className="flex items-center justify-between">
          <div className="mr-1 inline-flex grow">
            <span className="mr-2 text-sky-500">—</span>{" "}
            <a className="text-sm font-[650]">Python</a>
          </div>
          <div
            className={`relative h-1.5 w-60 shrink-0 bg-slate-200 before:absolute before:inset-0 before:w-[90%] before:bg-sky-500 dark:bg-slate-700 sm:w-80 md:w-20 lg:w-24`}
            role="progressbar"
            aria-valuenow={90}
            aria-valuemin={0}
            aria-valuemax={100}
          ></div>
        </li>
        <li className="flex items-center justify-between">
          <div className="mr-1 inline-flex grow">
            <span className="mr-2 text-sky-500">—</span>{" "}
            <a className="text-sm font-[650]">Bash/Shell</a>
          </div>
          <ProgressBar
            className="w-60 shrink-0 sm:w-80 md:w-20 lg:w-24"
            value={60}
          />
        </li>
        <li className="flex items-center justify-between">
          <div className="mr-1 inline-flex grow">
            <span className="mr-2 text-sky-500">—</span>{" "}
            <a className="text-sm font-[650]">Machine Learning</a>
          </div>
          <ProgressBar
            className="w-60 shrink-0 sm:w-80 md:w-20 lg:w-24"
            value={60}
          />
        </li>
        <li className="flex items-center justify-between">
          <div className="mr-1 inline-flex grow">
            <span className="mr-2 text-sky-500">—</span>{" "}
            <a className="text-sm font-[650]">Data Analytics</a>
          </div>
          <ProgressBar
            className="w-60 shrink-0 sm:w-80 md:w-20 lg:w-24"
            value={60}
          />
        </li>
        <li className="flex items-center justify-between">
          <div className="mr-1 inline-flex grow">
            <span className="mr-2 text-sky-500">—</span>{" "}
            <a className="text-sm font-[650]">Go</a>
          </div>
          <ProgressBar
            className="w-60 shrink-0 sm:w-80 md:w-20 lg:w-24"
            value={25}
          />
        </li>
      </ul>
    </section>
  );
}

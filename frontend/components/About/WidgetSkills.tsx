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
          <ProgressBar
            className="w-60 shrink-0 sm:w-80 md:w-20 lg:w-24"
            value="advanced"
          />
        </li>
        <li className="flex items-center justify-between">
          <div className="mr-1 inline-flex grow">
            <span className="mr-2 text-sky-500">—</span>{" "}
            <a className="text-sm font-[650]">Bash/Shell</a>
          </div>
          <ProgressBar
            className="w-60 shrink-0 sm:w-80 md:w-20 lg:w-24"
            value="intermediate"
          />
        </li>
        <li className="flex items-center justify-between">
          <div className="mr-1 inline-flex grow">
            <span className="mr-2 text-sky-500">—</span>{" "}
            <a className="text-sm font-[650]">Machine Learning</a>
          </div>
          <ProgressBar
            className="w-60 shrink-0 sm:w-80 md:w-20 lg:w-24"
            value="intermediate"
          />
        </li>
        <li className="flex items-center justify-between">
          <div className="mr-1 inline-flex grow">
            <span className="mr-2 text-sky-500">—</span>{" "}
            <a className="text-sm font-[650]">Data Analytics</a>
          </div>
          <ProgressBar
            className="w-60 shrink-0 sm:w-80 md:w-20 lg:w-24"
            value="intermediate"
          />
        </li>
        <li className="flex items-center justify-between">
          <div className="mr-1 inline-flex grow">
            <span className="mr-2 text-sky-500">—</span>{" "}
            <a className="text-sm font-[650]">Go</a>
          </div>
          <ProgressBar
            className="w-60 shrink-0 sm:w-80 md:w-20 lg:w-24"
            value="beginner"
          />
        </li>
      </ul>
    </section>
  );
}

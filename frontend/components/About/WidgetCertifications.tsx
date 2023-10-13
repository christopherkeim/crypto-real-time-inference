// IBM Applied DevOps Engineering Professional (IBM July 2023)
// DevOps and Software Engineering Professional (IBM July 2023)
// Machine Learning Specialization (Stanford University, DeepLearning.AI June 2023)
// CompTIA A+ Certification (CompTIA June 2023)
export function WidgetCertifications() {
  return (
    <section className="rounded-lg border border-slate-200 p-5 dark:border-slate-800 dark:bg-gradient-to-t dark:from-slate-800 dark:to-slate-800/30">
      <div className="mb-3 font-[650]">Certifications</div>
      <ul className="space-y-3">
        <li className="flex flex-col pb-2">
          <div className="mb-2">
            <p className="text-sm font-[650]">
              IBM Applied DevOps Engineering Professional
              <span className="mx-2 text-sky-500">—</span>
              IBM, July 2023
            </p>
          </div>
          <hr className="h-px border-0 bg-sky-200 dark:bg-sky-500" />
        </li>
        <li className="flex flex-col pb-2">
          <div className="mb-2">
            <p className="text-sm font-[650]">
              DevOps and Software Engineering Professional
              <span className="mx-2 text-sky-500">—</span>
              IBM, July 2023
            </p>
          </div>
          <hr className="h-px border-0 bg-sky-200 dark:bg-sky-500" />
        </li>
        <li className="flex flex-col pb-2">
          <div className="mb-2">
            <p className="text-sm font-[650]">
              Machine Learning Specialization
              <span className="mx-2 text-sky-500">—</span>
              Stanford University, DeepLearning.AI, June 2023
            </p>
          </div>
          <hr className="h-px border-0 bg-sky-200 dark:bg-sky-500" />
        </li>
        <li className="flex flex-col pb-2">
          <div className="mb-2">
            <p className="text-sm font-[650]">
              CompTIA A+ Certification
              <span className="mx-2 text-sky-500">—</span>
              CompTIA, June 2023
            </p>
          </div>
          <hr className="h-px border-0 bg-sky-200 dark:bg-sky-500" />
        </li>
      </ul>
    </section>
  );
}

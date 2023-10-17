import Image from "next/image";

import UniversityOfVermont from "@/public/images/university_of_vermont_logo.png";
import CopenhagenUniversityLogo from "@/public/images/copenhagen_university_logo.svg";

export function Education() {
  return (
    <section className="space-y-8">
      <h4 className="h4 font-aspekta text-slate-800 dark:text-slate-100">
        Education
      </h4>
      <ul className="space-y-8">
        {/* Item */}
        <li className="group relative">
          <div className="flex items-start before:absolute before:left-0 before:ml-[28px] before:h-full before:w-px before:-translate-x-1/2 before:translate-y-8 before:self-start before:bg-slate-200 before:group-last-of-type:hidden before:dark:bg-slate-800">
            <div className="absolute left-0 flex h-14 w-14 items-center justify-center rounded-full border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 dark:bg-gradient-to-t dark:from-slate-800 dark:to-slate-800/30">
              <Image
                src={UniversityOfVermont}
                width={24}
                height={24}
                alt="University of Vermont"
              />
            </div>
            <div className="space-y-1 pl-20">
              <div className="text-xs uppercase text-slate-500">
                Aug 2014{" "}
                <span className="text-slate-400 dark:text-slate-600">·</span>{" "}
                May 2018
              </div>
              <div className="font-aspekta font-[650] text-slate-800 dark:text-slate-100">
                Bachelor of Science - Neuroscience
              </div>
              <div className="text-sm font-medium text-slate-800 dark:text-slate-100">
                University of Vermont
              </div>
            </div>
          </div>
        </li>
        <li className="group relative">
          <div className="flex items-start before:absolute before:left-0 before:ml-[28px] before:h-full before:w-px before:-translate-x-1/2 before:translate-y-8 before:self-start before:bg-slate-200 before:group-last-of-type:hidden before:dark:bg-slate-800">
            <div className="absolute left-0 flex h-14 w-14 items-center justify-center rounded-full border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 dark:bg-gradient-to-t dark:from-slate-800 dark:to-slate-800/30">
              <Image
                src={CopenhagenUniversityLogo}
                width={24}
                height={26}
                alt="Copenhagen University"
              />
            </div>
            <div className="space-y-1 pl-20">
              <div className="text-xs uppercase text-slate-500">Dec 2017</div>
              <div className="font-aspekta font-[650] text-slate-800 dark:text-slate-100">
                Study Abroad Psychopharmacology
              </div>
              <div className="text-sm font-medium text-slate-800 dark:text-slate-100">
                Copenhag University
              </div>
            </div>
          </div>
        </li>
      </ul>
    </section>
  );
}

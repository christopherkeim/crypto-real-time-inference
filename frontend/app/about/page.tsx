export const metadata = {
  title: "About - Appy",
  description: "Page description",
};

import Image from "next/image";
import ChristopherKeim from "@/public/images/christopher_keim.jpg";
import { FadeAnimation } from "@/components/ui/FadeAnimation";
import { Education } from "@/components/About/Education";
import { Experience } from "@/components/About/Experience";
import { WidgetSkills } from "@/components/About/WidgetSkills";
import { WidgetCertifications } from "@/components/About/WidgetCertifications";

export default function About() {
  return (
    <div className="relative">
      <div className="relative mx-auto max-w-6xl px-4 sm:px-6">
        <div className="pb-12 pt-32 md:pb-20 ">
          <div className="grow space-y-8 pb-16 pt-12 md:flex md:space-x-8 md:space-y-0 md:pb-20 md:pt-16">
            {/* Middle area */}
            <div className="grow">
              <div className="max-w-[700px]">
                <section>
                  {/* Page title */}
                  <FadeAnimation>
                    <h2 className="h2 mb-4">Christopher Keim</h2>
                    <div className="relative mb-4 flex min-h-[500px] min-w-full shrink-0 grow overflow-hidden">
                      <Image
                        src={ChristopherKeim}
                        alt="Christopher Keim"
                        fill
                        className="h-full w-full rounded-lg"
                        style={{ objectFit: "cover", objectPosition: "center" }}
                      />
                    </div>
                    <h3 className="h4 mb-12">
                      Machine Learning Engineer and Published Neuroscience
                      Researcher
                    </h3>
                  </FadeAnimation>
                  {/* Page content */}
                  <div className="space-y-12 text-slate-500 dark:text-slate-400">
                    <FadeAnimation>
                      <Education />
                    </FadeAnimation>
                    <FadeAnimation>
                      <Experience />
                    </FadeAnimation>
                  </div>
                </section>
              </div>
            </div>

            {/* Right sidebar */}
            <aside className="shrink-0 md:w-[240px] lg:w-[300px]">
              <div className="space-y-6">
                <FadeAnimation fadeDelay={200}>
                  <WidgetSkills />
                </FadeAnimation>
                <FadeAnimation fadeDelay={400}>
                  <WidgetCertifications />
                </FadeAnimation>
              </div>
            </aside>
          </div>
        </div>
      </div>
    </div>
  );
}

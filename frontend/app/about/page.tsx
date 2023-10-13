export const metadata = {
  title: "About - Appy",
  description: "Page description",
};

// import Hero from "@/components/hero-about";
// import FeaturesGallery from "@/components/features-gallery";
// import Timeline from "@/components/timeline";
// import Career from "@/components/career";
// import Team from "@/components/team";
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
                  <h2 className="h2 mb-2">Christopher Keim</h2>
                  <h3 className="h4 mb-12">
                    Machine Learning Engineer and Published Neuroscience
                    Researcher
                  </h3>
                  {/* Page content */}
                  <div className="space-y-12 text-slate-500 dark:text-slate-400">
                    <Education />
                    <Experience />
                    {/* <Awards />
              <Recommendations /> */}
                  </div>
                </section>
              </div>
            </div>

            {/* Right sidebar */}
            <aside className="shrink-0 md:w-[240px] lg:w-[300px]">
              <div className="space-y-6">
                <WidgetSkills />
                <WidgetCertifications />
              </div>
            </aside>
          </div>
        </div>
      </div>
    </div>
  );
}
// <section>
//   <h2 id="christopher-keim-tiny-circle-photo-from-bottom-">
//     Christopher Keim [tiny circle photo from bottom]
//   </h2>
//   <h3 id="interested-in-data-science-machine-learning-devops-and-backend-roles-">
//     Interested in <strong>Data Science</strong>,{" "}
//     <strong>Machine Learning</strong>, <strong>DevOps</strong> and{" "}
//     <strong>Backend</strong> roles.
//   </h3>
//   <h3 id="self-taught-machine-learning-engineer-and-published-neuroscience-researcher-">
//     Self-taught Machine Learning Engineer and published Neuroscience
//     researcher.
//   </h3>
//

//     Open to remote work. Willing to Relocate to: New York City, San
//     Francisco Bay Area Authorized to work in the US
//   </p>

//   {/* <Hero />
//   <FeaturesGallery />
//   <Timeline />
//   <Career />
//   <Team /> */}
// </section>

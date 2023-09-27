export const metadata = {
  title: "About - Appy",
  description: "Page description",
};

import Hero from "@/components/hero-about";
import FeaturesGallery from "@/components/features-gallery";
import Timeline from "@/components/timeline";
import Career from "@/components/career";
import Team from "@/components/team";

export default function About() {
  return (
    <>
      <Hero />
      <FeaturesGallery />
      <Timeline />
      <Career />
      <Team />
    </>
  );
}

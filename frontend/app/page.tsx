export const metadata = {
  title: "Home - Appy",
  description: "Page description",
};

import PageIllustration from "@/components/page-illustration";
import Hero from "@/components/hero-home";
import Stats from "@/components/stats";
import Carousel from "@/components/carousel";
import Tabs from "@/components/tabs";
import Process from "@/components/process";
import PricingTables from "@/components/pricing-tables";
import TestimonialsBlocks from "@/components/testimonials-blocks";
import FeaturesBlocks from "@/components/features-blocks";
import Cta from "@/components/cta";

export default function Home() {
  return (
    <>
      <Hero />
      <Stats />
      <Carousel />
      <Tabs />
      <Process />
      <PricingTables />
      <TestimonialsBlocks />
      <FeaturesBlocks />
      <Cta />
    </>
  );
}

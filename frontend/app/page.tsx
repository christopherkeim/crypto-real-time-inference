import Hero from "@/components/hero-home";
import { ResearchPaper } from "@/components/ResearchPaper/ResearchPaper";

export const metadata = {
  title: "Crypto Real Time Inference",
  description: "Built by Chris Keim",
};

export default function Home() {
  return (
    <>
      <Hero />
      <ResearchPaper />
    </>
  );
}

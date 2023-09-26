import { PredictionForm } from "./PredictionForm";

export default function HeroHome() {
  return (
    <section>
      <div className="max-w-6xl mx-auto px-4 sm:px-6 md:px-12 lg:px-20 pt-32 pb-10 md:pt-40 md:pb-16 flex flex-col md:gap-12 lg:gap-20 items-center mb-8 text-left">
        <h1
          className="h1 lg:text-6xl font-red-hat-display font-black"
          data-aos="fade-down"
        >
          Crypto <span className="text-blue-500">Real Time</span>{" "}
          <span className="text-green-500">Inference</span>
        </h1>
        <PredictionForm />
        <p
          className="text-xl text-gray-600 dark:text-gray-400"
          data-aos="fade-down"
          data-aos-delay="150"
        >
          Our landing page template works on all devices, so you only have to
          set it up once, and get beautiful results forever.
        </p>
      </div>
    </section>
  );
}

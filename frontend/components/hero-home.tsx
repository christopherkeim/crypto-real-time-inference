import { PredictionContainer } from "./Prediction/PredictionContainer";

export default function HeroHome() {
  return (
    <section>
      <div className="mx-auto mb-8 flex max-w-6xl flex-col items-center px-4 pb-10 pt-32 text-left sm:px-6 md:gap-12 md:px-12 md:pb-16 md:pt-40 lg:gap-20 lg:px-20">
        <h1
          className="h1 font-red-hat-display font-black lg:text-6xl"
          data-aos="fade-down"
        >
          Crypto <span className="text-blue-500">Real Time</span>{" "}
          <span className="text-green-500">Inference</span>
        </h1>
        <PredictionContainer />
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

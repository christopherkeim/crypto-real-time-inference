import Image from "next/image";
import HeroImage from "@/public/images/about-hero.jpg";

export default function HeroAbout() {
  return (
    <section className="relative">
      {/* Background image */}

      <div className="relative max-w-6xl mx-auto px-4 sm:px-6">
        <div className="pt-32 pb-12 md:pt-40 md:pb-20">
          <div className="text-center">
            <div className="relative flex justify-center items-center">
              <div
                className="relative inline-flex items-start"
                data-aos="fade-up"
              >
                <Image
                  className="opacity-50"
                  src={HeroImage}
                  width={768}
                  height={432}
                  priority
                  alt="About hero"
                />
                <div
                  className="absolute inset-0 bg-gradient-to-t from-white dark:from-gray-900"
                  aria-hidden="true"
                ></div>
              </div>
              <div className="absolute" data-aos="fade-down">
                <h1 className="h1 lg:text-6xl font-red-hat-display">
                  Make your own <span className="text-blue-500">way</span>
                </h1>
              </div>
              <div
                className="absolute bottom-0 -mb-8 w-0.5 h-16 bg-gray-300 dark:bg-gray-700"
                aria-hidden="true"
              ></div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

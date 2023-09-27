import Link from "next/link";
import Image from "next/image";
import NotFoundImage from "@/public/images/404.jpg";

export default function NotFound() {
  return (
    <>
      <section className="relative">
        <div className="max-w-6xl mx-auto px-4 sm:px-6">
          <div className="pt-32 pb-12 md:pt-40 md:pb-20">
            <div className="max-w-3xl mx-auto text-center">
              <div className="relative inline-flex justify-center items-center">
                <Image
                  className="hidden sm:block opacity-50 md:opacity-80"
                  src={NotFoundImage}
                  width={768}
                  height={432}
                  priority
                  alt="404"
                />
                <div
                  className="hidden sm:block absolute inset-0 bg-gradient-to-t from-white dark:from-gray-900"
                  aria-hidden="true"
                ></div>
                <div className="sm:absolute w-full">
                  <h1 className="h3 font-red-hat-display mb-8">
                    Hm, the page you were looking for doesn't exist anymore.
                  </h1>
                  <Link
                    className="btn text-white bg-blue-500 hover:bg-blue-400 inline-flex items-center"
                    href="/"
                  >
                    <span>Back to Home</span>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}

export default function Footer() {
  return (
    <footer className="relative">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="-mt-px border-t border-gray-200 py-12 dark:border-gray-800 md:py-16">
          <div className="mr-4 text-sm text-gray-600 dark:text-gray-400">
            {" "}
            <a
              href="https://github.com/christopherkeim/crypto-real-time-inference"
              target="_blank"
              referrerPolicy="no-referrer"
              className="text-blue-600 transition duration-150 ease-in-out hover:underline dark:text-blue-100"
            >
              Source Code
            </a>
            &nbsp;|&nbsp;
            <a
              className="text-blue-600 transition duration-150 ease-in-out hover:underline dark:text-blue-100"
              href="https://github.com/christopherkeim/crypto-real-time-inference/blob/main/LICENSE"
              target="_blank"
              referrerPolicy="no-referrer"
            >
              License
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}

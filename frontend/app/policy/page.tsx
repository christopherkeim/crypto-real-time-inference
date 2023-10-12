import FadeAnimation from "@/components/ui/FadeAnimation";

export default function page() {
  return (
    <section className="relative">
      <div className="relative mx-auto max-w-6xl px-4 sm:px-6">
        <div className="pb-12 pt-32 md:pb-20 md:pt-40">
          <FadeAnimation className="mb-10 text-center">
            <h1 className="h1 font-red-hat-display">Policy</h1>
          </FadeAnimation>
          <FadeAnimation className="mb-5" fadeDelay={100}>
            <h2 className="h2 font-red-hat-display">Disclaimer</h2>
          </FadeAnimation>
          <FadeAnimation className="mb-10" fadeDelay={200}>
            <p>
              <strong>
                Cryptocurrency trading involves inherent risks and is subject to
                market fluctuations. The code here is intended for informational
                purposes only and should not be considered financial advice.
                Always conduct thorough research and exercise caution when
                trading cryptocurrencies.
              </strong>
            </p>
          </FadeAnimation>
          <FadeAnimation className="mb-2" fadeDelay={300}>
            <h2 className="h2 font-red-hat-display">MIT License</h2>
          </FadeAnimation>
          <FadeAnimation className="mb-5" fadeDelay={400}>
            <p>
              <strong>Copyright (c) 2023 Christopher Keim</strong>
            </p>
          </FadeAnimation>
          <FadeAnimation className="mb-5" fadeDelay={500}>
            <p>
              Permission is hereby granted, free of charge, to any person
              obtaining a copy of this software and associated documentation
              files (the &quot;Software&quot;), to deal in the Software without
              restriction, including without limitation the rights to use, copy,
              modify, merge, publish, distribute, sublicense, and/or sell copies
              of the Software, and to permit persons to whom the Software is
              furnished to do so, subject to the following conditions:
            </p>
          </FadeAnimation>
          <FadeAnimation className="mb-5" fadeDelay={600}>
            <p>
              The above copyright notice and this permission notice shall be
              included in all copies or substantial portions of the Software.
            </p>
          </FadeAnimation>
          <FadeAnimation className="mb-2" fadeDelay={700}>
            <p>
              <strong>
                THE SOFTWARE IS PROVIDED &quot;AS IS&quot;, WITHOUT WARRANTY OF
                ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
                WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
                AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
                HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
                WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
                FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
                OTHER DEALINGS IN THE SOFTWARE.
              </strong>
            </p>
          </FadeAnimation>
        </div>
      </div>
    </section>
  );
}

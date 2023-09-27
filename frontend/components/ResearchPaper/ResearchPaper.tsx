import { PaperHeader } from "./PaperHeader";
import { PaperImage } from "./PaperImage";
import { PaperParagraph } from "./PaperParagraph";

export function ResearchPaper() {
  return (
    <section>
      <div className="mx-auto my-2 max-w-6xl px-4 sm:px-6">
        {/* Paper effect */}
        <div className="relative rounded-sm bg-gray-50 px-8 py-10 dark:bg-zinc-900 md:px-12 md:py-10">
          <div className="relative mx-auto max-w-3xl ">
            <PaperHeader>Reaserch Paper Title</PaperHeader>
            <PaperParagraph>
              Do not worry too about picking exactly the right role; we can
              always give you more options after starting the conversation.
            </PaperParagraph>
            <PaperImage src="/images/404.jpg" />
          </div>
        </div>
      </div>
    </section>
  );
}

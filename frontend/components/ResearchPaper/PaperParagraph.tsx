type PaperParagraphProps = {
  children: React.ReactNode;
  textSize?: "md" | "lg" | "xl";
};

export function PaperParagraph({
  children,
  textSize = "md",
}: PaperParagraphProps) {
  return (
    <p className={`mb-2 text-gray-600 dark:text-gray-400 text-${textSize}`}>
      {children}
    </p>
  );
}

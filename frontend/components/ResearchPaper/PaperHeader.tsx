type PaperHeaderProps = {
  children: React.ReactNode;
  headerLevel?: 1 | 2 | 3;
};

export function PaperHeader({ children, headerLevel = 3 }: PaperHeaderProps) {
  return (
    <h3
      className={`h${headerLevel} text-gray-700 font-red-hat-display dark:text-gray-100 mb-2`}
    >
      {children}
    </h3>
  );
}

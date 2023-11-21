import Link from "next/link";
import ThemeToggle from "./theme-toggle";

export default function Header() {
  return (
    <header className="absolute z-30 w-full">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4">
        <div className="ms-3 mt-2 flex items-end gap-6">
          <Link href={"/"}>
            <h4 className="h4 border-b">Home</h4>
          </Link>
          <Link href={"/policy"}>
            <h5 className="h5 border-b">Policy</h5>
          </Link>
        </div>
        <ThemeToggle className="ms-14 mt-3" />
      </div>
    </header>
  );
}

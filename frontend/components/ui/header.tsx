import Link from "next/link";
import Logo from "./logo";
import ThemeToggle from "./theme-toggle";
import MobileMenu from "./mobile-menu";

export default function Header() {
  return (
    <header className="absolute z-30 w-full">
      <div className="mx-auto flex max-w-6xl justify-start px-4">
        <ThemeToggle className="ms-14 mt-3" />
      </div>
    </header>
  );
}

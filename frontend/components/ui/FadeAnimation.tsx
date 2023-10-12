// This component is used to fade in elements on the page.
// It uses the AOS library to do this.
// It is implied that AOS is already installed and configured in the project.

import React from "react";

type FadeAnimationProps = {
  className?: string;
  fadeDirection?: "up" | "down" | "left" | "right";
  fadeDelay?: number;
  children?: React.ReactNode;
};

export function FadeAnimation({
  className,
  fadeDirection = "up",
  fadeDelay = 0,
  children,
}: FadeAnimationProps) {
  return (
    <div
      className={className ?? ""}
      data-aos={`fade-${fadeDirection}`}
      data-aos-delay={fadeDelay}
    >
      {children}
    </div>
  );
}

export default FadeAnimation;

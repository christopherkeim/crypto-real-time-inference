import React, { useEffect } from "react";
import AOS from "aos";
import "aos/dist/aos.css";

interface FadeAnimationProps {
  className?: string;
  fadeDirection?: "up" | "down" | "left" | "right";
  fadeDelay?: number;
  children?: React.ReactNode;
}

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

export default function Logo() {
  return (
    <svg className="w-8 h-8" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient x1="26%" y1="100%" x2="100%" y2="100%" id="logo_a">
          <stop stopColor="#3ABAB4" offset="0%" />
          <stop stopColor="#7F9CF5" offset="100%" />
        </linearGradient>
        <linearGradient x1="26%" y1="100%" x2="100%" y2="100%" id="logo_b">
          <stop stopColor="#3ABAB4" offset="0%" />
          <stop stopColor="#3ABAB4" stopOpacity="0" offset="100%" />
        </linearGradient>
      </defs>
      <path d="M32 16h-8a8 8 0 10-16 0H0C0 7.163 7.163 0 16 0s16 7.163 16 16z" fill="url(#logo_a)" />
      <path d="M32 16c0 8.837-7.163 16-16 16S0 24.837 0 16h8a8 8 0 1016 0h8z" fill="url(#logo_b)" />
    </svg>
  )
}

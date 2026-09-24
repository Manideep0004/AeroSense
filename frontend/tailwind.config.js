/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        "primary": "#06b6d4",
        "secondary": "#8b5cf6",
        "tertiary": "#10b981",
        "error": "#f43f5e",
        "warning": "#f59e0b",
      },
      fontFamily: {
        sans: ['Geist', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
        display: ['"Space Grotesk"', 'sans-serif']
      }
    },
  },
  plugins: [],
}

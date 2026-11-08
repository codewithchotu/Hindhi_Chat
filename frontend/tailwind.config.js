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
        dark: {
          bg: '#121214',
          card: '#1e1e24',
          border: '#2e2e38',
          text: '#ececf1',
          muted: '#9ca3af'
        },
        light: {
          bg: '#f9fafb',
          card: '#ffffff',
          border: '#e5e7eb',
          text: '#111827',
          muted: '#6b7280'
        }
      }
    },
  },
  plugins: [],
}

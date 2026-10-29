import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: { 50: '#effaf8', 100: '#d3f3ee', 500: '#14a090', 600: '#0f857a', 700: '#0f6b63', 900: '#0b3f3b' },
      },
    },
  },
  plugins: [],
};
export default config;

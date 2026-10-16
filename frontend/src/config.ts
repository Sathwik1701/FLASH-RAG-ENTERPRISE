// API Base URL config: supports VITE_API_BASE_URL for Vercel/Production
// Defaults to empty string in development so Vite proxy handles /api
// Strips any accidental trailing slash to ensure clean request interpolation
const rawBaseUrl = import.meta.env.VITE_API_BASE_URL || '';
export const API_BASE_URL = rawBaseUrl.endsWith('/') ? rawBaseUrl.slice(0, -1) : rawBaseUrl;

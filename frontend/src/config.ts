// API Base URL config: supports VITE_API_BASE_URL for Vercel/Production
// Defaults to empty string in development so Vite proxy handles /api
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

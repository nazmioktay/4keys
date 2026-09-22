// Backend'e (FastAPI) ince bir fetch sarmalayıcısı.
// Geliştirmede varsayılan olarak localhost:8000'i hedefler; farklı bir
// backend adresi için .env dosyasına VITE_API_BASE_URL yazabilirsiniz.
const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const TOKEN_KEY = "4keys_token";

const auth = {
  getToken: () => localStorage.getItem(TOKEN_KEY),
  setToken: (token) => localStorage.setItem(TOKEN_KEY, token),
  clearToken: () => localStorage.removeItem(TOKEN_KEY),
};

async function request(path, options = {}) {
  const token = auth.getToken();
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    ...options,
  });
  if (response.status === 401 && path !== "/auth/login") {
    // Token yok/geçersiz/süresi dolmuş — backend'in TÜM uç noktaları
    // AuthMiddleware arkasında olduğu için bu her istekte olabilir.
    // Oturumu temizleyip giriş sayfasına at.
    auth.clearToken();
    if (!window.location.pathname.startsWith("/login")) {
      window.location.href = "/login";
    }
    throw new Error("Oturum sona erdi, tekrar giriş yapın.");
  }
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const detail = (data && (data.detail || data.message)) || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

export const api = {
  get: (path) => request(path),
  post: (path, body) => request(path, { method: "POST", body: JSON.stringify(body) }),
  put: (path, body) => request(path, { method: "PUT", body: JSON.stringify(body) }),
};

export { BASE_URL, auth };

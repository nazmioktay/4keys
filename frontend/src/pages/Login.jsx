import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, auth } from "../api.js";
import ErrorBanner from "../components/ErrorBanner.jsx";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await api.post("/auth/login", { username, password });
      auth.setToken(res.token);
      navigate("/", { replace: true });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page" style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh" }}>
      <form className="card" style={{ width: "100%", maxWidth: 360 }} onSubmit={submit}>
        <div className="card-title">4keys — Giriş</div>

        <label className="field">Kullanıcı adı</label>
        <input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus autoComplete="username" />

        <label className="field">Şifre</label>
        <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" />

        <div style={{ height: 14 }} />
        <button className="primary" type="submit" disabled={loading || !username || !password}>
          {loading ? "Giriş yapılıyor..." : "Giriş yap"}
        </button>

        <ErrorBanner message={error} />
      </form>
    </div>
  );
}

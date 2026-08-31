import { useState } from "react";
import { api } from "../api.js";
import ErrorBanner from "../components/ErrorBanner.jsx";
import Loading from "../components/Loading.jsx";

export default function Screener() {
  const [direction, setDirection] = useState("long");
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const load = async (dir) => {
    setDirection(dir);
    setLoading(true);
    setError("");
    try {
      const res = await api.get(`/screener/top?direction=${dir}&limit=10`);
      setResults(res);
    } catch (err) {
      setError(err.message);
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <h1 className="page-title">Araştırıcı</h1>
      <div className="tabs">
        <button className={"tab-btn" + (direction === "long" ? " active" : "")} onClick={() => load("long")}>
          Top 10 Long
        </button>
        <button className={"tab-btn" + (direction === "short" ? " active" : "")} onClick={() => load("short")}>
          Top 10 Short
        </button>
      </div>

      <ErrorBanner message={error} />
      {loading && <Loading label="Taranıyor..." />}

      {!loading && !results && !error && (
        <div className="empty-state">Bir yön seçip taramayı başlatın.</div>
      )}

      {results?.length === 0 && <div className="empty-state">Sonuç bulunamadı.</div>}

      {results?.map((r) => (
        <div className="card" key={r.symbol}>
          <div className="row">
            <div>
              <div className="row-value">{r.symbol}</div>
              <div className="muted">RSI {r.rsi.toFixed(1)} · {r.trend === "up" ? "Yükseliş" : "Düşüş"}</div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div className="row-value">${r.close}</div>
              <div className={r.score >= 0 ? "pos" : "neg"}>{r.score.toFixed(1)} skor</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

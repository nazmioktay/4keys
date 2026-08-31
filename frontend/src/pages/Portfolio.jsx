import { useEffect, useState } from "react";
import { api } from "../api.js";
import Loading from "../components/Loading.jsx";
import ErrorBanner from "../components/ErrorBanner.jsx";

function fmt(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  return n.toLocaleString("tr-TR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export default function Portfolio() {
  const [status, setStatus] = useState(null);
  const [security, setSecurity] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setError("");
    try {
      const [p, s] = await Promise.all([api.get("/portfolio/status"), api.get("/security/status")]);
      setStatus(p);
      setSecurity(s);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  if (loading) return <div className="page"><Loading /></div>;

  const pnl = status ? status.equity - status.starting_equity : 0;
  const pnlPct = status && status.starting_equity ? (pnl / status.starting_equity) * 100 : 0;
  const isUp = pnl >= 0;

  return (
    <div className="page">
      <h1 className="page-title">Portföy</h1>
      <ErrorBanner message={error} />

      {security?.kill_switch?.active && (
        <div className="banner error">
          🛑 Kill switch aktif — yeni pozisyon açılmıyor. Sebep: {security.kill_switch.reason}
        </div>
      )}

      {status && (
        <div className="card">
          <div className="card-title">Toplam değer</div>
          <div className="value-lg">${fmt(status.equity)}</div>
          <div className={isUp ? "pos" : "neg"}>
            {isUp ? "+" : ""}${fmt(pnl)} · {isUp ? "+" : ""}{fmt(pnlPct)}% (oturum)
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-title">
          Açık pozisyonlar
          <span className="pill">{status?.open_positions?.length ?? 0}</span>
        </div>
        {!status?.open_positions?.length && <div className="muted">Şu an açık pozisyon yok.</div>}
        {status?.open_positions?.map((p) => (
          <div className="row" key={p.symbol}>
            <div>
              <div className="row-value">{p.symbol}</div>
              <div className="muted">{p.direction === "long" ? "Long" : "Short"} · ${fmt(p.entry_price)}</div>
            </div>
            <div className="row-value">${fmt(p.size_quote)}</div>
          </div>
        ))}
      </div>

      <div className="card">
        <div className="card-title">İşlem istatistikleri</div>
        <div className="row">
          <span className="row-label">Toplam işlem</span>
          <span className="row-value">{status?.trade_stats?.num_trades ?? 0}</span>
        </div>
        <div className="row">
          <span className="row-label">Kazanma oranı</span>
          <span className="row-value">{fmt(status?.trade_stats?.win_rate_pct)}%</span>
        </div>
        <div className="row">
          <span className="row-label">Ort. kazanç</span>
          <span className="row-value pos">+{fmt(status?.trade_stats?.avg_win_pct)}%</span>
        </div>
        <div className="row">
          <span className="row-label">Ort. kayıp</span>
          <span className="row-value neg">{fmt(status?.trade_stats?.avg_loss_pct)}%</span>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Son kapanan işlemler</div>
        {!status?.closed_history?.length && <div className="muted">Henüz kapanan işlem yok.</div>}
        {status?.closed_history?.slice().reverse().slice(0, 8).map((t, i) => (
          <div className="row" key={i}>
            <span className="row-label">{t.symbol}</span>
            <span className={"row-value " + (t.pnl_pct >= 0 ? "pos" : "neg")}>
              {t.pnl_pct >= 0 ? "+" : ""}{fmt(t.pnl_pct)}%
            </span>
          </div>
        ))}
      </div>

      <div className="card">
        <div className="card-title">Risk kuralları</div>
        <div className="row">
          <span className="row-label">Boyutlandırma</span>
          <span className="row-value">{status?.rules?.position_sizing_method === "kelly" ? "Kelly" : "Sabit risk"}</span>
        </div>
        <div className="row">
          <span className="row-label">İşlem başına risk</span>
          <span className="row-value">%{status?.rules?.max_risk_per_trade_pct}</span>
        </div>
        <div className="row">
          <span className="row-label">Günlük zarar limiti</span>
          <span className="row-value">%{status?.rules?.daily_loss_limit_pct}</span>
        </div>
      </div>
    </div>
  );
}

import { useEffect, useState } from "react";
import { api } from "../api.js";
import Loading from "../components/Loading.jsx";
import ErrorBanner from "../components/ErrorBanner.jsx";
import RealPositionsCard from "../components/RealPositionsCard.jsx";

function fmt(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  return Number(n).toLocaleString("tr-TR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export default function Portfolio() {
  const [security, setSecurity] = useState(null);
  const [balance, setBalance] = useState(null);
  const [balanceError, setBalanceError] = useState("");
  const [pnl, setPnl] = useState(null);
  const [pnlError, setPnlError] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setError("");
    setBalanceError("");
    setPnlError("");
    try {
      const s = await api.get("/security/status");
      setSecurity(s);
    } catch (err) {
      setError(err.message);
    }
    try {
      const b = await api.get("/trading/balance?market_type=future");
      setBalance(b);
    } catch (err) {
      setBalanceError(err.message);
    } finally {
      setLoading(false);
    }
    try {
      setPnl(await api.get("/trading/pnl-summary"));
    } catch (err) {
      setPnlError(err.message);
    }
  };

  useEffect(() => {
    load();
  }, []);

  if (loading) return <div className="page"><Loading /></div>;

  const info = balance?.info || {};
  const walletBalance = Number(info.totalWalletBalance ?? balance?.total?.USDT ?? 0);
  const unrealizedPnl = Number(info.totalUnrealizedProfit ?? 0);
  const marginBalance = Number(info.totalMarginBalance ?? walletBalance + unrealizedPnl);
  const availableBalance = Number(info.availableBalance ?? 0);
  const isUp = unrealizedPnl >= 0;
  const gatesOpen = security?.live_trading_enabled && !security?.kill_switch?.active;

  return (
    <div className="page">
      <h1 className="page-title">Portföy</h1>
      <p className="muted">Gerçek Binance hesabı (Futures) — paper trading/otopilot simülasyonları ayrı sekmelerde.</p>
      <ErrorBanner message={error} />
      <ErrorBanner message={balanceError} />

      {security?.kill_switch?.active && (
        <div className="banner error">
          🛑 Kill switch aktif — yeni pozisyon açılmıyor. Sebep: {security.kill_switch.reason}
        </div>
      )}

      <div className="card">
        <div className="card-title">Toplam değer (marjin bakiyesi)</div>
        <div className="value-lg">${fmt(marginBalance)}</div>
        <div className={isUp ? "pos" : "neg"}>
          {isUp ? "+" : ""}${fmt(unrealizedPnl)} açık pozisyon P&L
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <span className="row-label">Cüzdan bakiyesi</span>
          <span className="row-value">${fmt(walletBalance)}</span>
        </div>
        <div className="row">
          <span className="row-label">Kullanılabilir bakiye</span>
          <span className="row-value">${fmt(availableBalance)}</span>
        </div>
      </div>

      <RealPositionsCard gatesOpen={gatesOpen} />

      <ErrorBanner message={pnlError} />
      <PnlSummaryCard pnl={pnl} />
    </div>
  );
}

function PnlSummaryCard({ pnl }) {
  if (!pnl) return null;
  const rows = [
    { label: "Bugün (son 24s)", w: pnl.daily },
    { label: "Bu hafta (son 7g)", w: pnl.weekly },
    { label: "Bu ay (son 30g)", w: pnl.monthly },
    { label: "Toplam", w: pnl.total },
  ];
  return (
    <div className="card">
      <div className="card-title">PNL özeti (gerçekleşmiş, gerçek hesap)</div>
      {rows.map((r) => (
        <div className="row" key={r.label}>
          <div>
            <div className="row-value">{r.label}</div>
            <div className="muted">{r.w.trade_count} kayıt · %{fmt(r.w.win_rate_pct)} kazanma</div>
          </div>
          <span className={"row-value " + (r.w.pnl_quote >= 0 ? "pos" : "neg")}>
            {r.w.pnl_quote >= 0 ? "+" : ""}${fmt(r.w.pnl_quote)}
          </span>
        </div>
      ))}
    </div>
  );
}

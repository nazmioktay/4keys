import { useEffect, useState } from "react";
import { api } from "../api.js";
import Loading from "../components/Loading.jsx";
import ErrorBanner from "../components/ErrorBanner.jsx";

function fmt(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  return n.toLocaleString("tr-TR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function fmtDate(iso) {
  if (!iso) return "-";
  return new Date(iso).toLocaleString("tr-TR", { dateStyle: "short", timeStyle: "short" });
}

const DEFAULT_PARAMS = {
  symbol: "BTC/USDT:USDT",
  timeframe: "1h",
  candles: 10000,
  initial_balance: 1000,
  open_confidence: 0.6,
  close_confidence: 0.55,
  stop_loss_pct: 3.0,
  commission_pct: 0.04,
  slippage_pct: 0.02,
  use_meta_label: true,
};

export default function Backtest() {
  const [report, setReport] = useState(null);
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  const loadLatest = async () => {
    setError("");
    try {
      const res = await api.get(`/backtest/system/latest?symbol=${encodeURIComponent(DEFAULT_PARAMS.symbol)}`);
      setReport(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadLatest();
  }, []);

  const runBacktest = async () => {
    setRunning(true);
    setError("");
    try {
      const res = await api.post("/backtest/system/run", params);
      setReport(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  };

  if (loading) return <div className="page"><Loading /></div>;

  return (
    <div className="page">
      <h1 className="page-title">Backtest</h1>
      <ErrorBanner message={error} />

      <ParamsCard params={params} setParams={setParams} onRun={runBacktest} running={running} />
      {report && <SummaryCard report={report} />}
      {report && report.warnings?.length > 0 && <WarningsCard warnings={report.warnings} />}
      {report && <TradesCard trades={report.trades} />}
      {!report && !running && <div className="card"><div className="muted">Henüz bir backtest çalıştırılmadı.</div></div>}
    </div>
  );
}

function ParamsCard({ params, setParams, onRun, running }) {
  const set = (key) => (e) => {
    const raw = e.target.value;
    const value = e.target.type === "checkbox" ? e.target.checked : e.target.type === "number" ? Number(raw) : raw;
    setParams({ ...params, [key]: value });
  };

  return (
    <div className="card">
      <div className="card-title">
        Sistem backtest parametreleri
        <button className="primary" style={{ marginLeft: "auto", padding: "6px 14px" }} onClick={onRun} disabled={running}>
          {running ? "Çalışıyor... (10.000 mum taranıyor)" : "Backtest Çalıştır"}
        </button>
      </div>
      <p className="muted">
        Canlı karar motorunun kullandığı AYNI eğitilmiş model (+ varsa meta-label filtresi), BTCUSDT.P futures üzerinde
        geçmiş {params.candles.toLocaleString("tr-TR")} mumu bar-bar tekrar oynatır. Sonuç Grafana'da da (candlestick +
        al/sat işaretleri + kümülatif equity) görüntülenebilir.
      </p>
      <div className="grid-2col">
        <div>
          <label className="field">Sembol</label>
          <input value={params.symbol} onChange={set("symbol")} />
        </div>
        <div>
          <label className="field">Zaman dilimi</label>
          <input value={params.timeframe} onChange={set("timeframe")} />
        </div>
        <div>
          <label className="field">Mum sayısı</label>
          <input type="number" value={params.candles} onChange={set("candles")} />
        </div>
        <div>
          <label className="field">Başlangıç bakiyesi ($)</label>
          <input type="number" value={params.initial_balance} onChange={set("initial_balance")} />
        </div>
        <div>
          <label className="field">Açılış güven eşiği</label>
          <input type="number" step="0.01" value={params.open_confidence} onChange={set("open_confidence")} />
        </div>
        <div>
          <label className="field">Kapanış güven eşiği</label>
          <input type="number" step="0.01" value={params.close_confidence} onChange={set("close_confidence")} />
        </div>
        <div>
          <label className="field">Stop-loss (%)</label>
          <input type="number" step="0.1" value={params.stop_loss_pct ?? ""} onChange={set("stop_loss_pct")} />
        </div>
        <div>
          <label className="field">Komisyon + kayma (%, işlem bacağı başına)</label>
          <input type="number" step="0.01" value={params.commission_pct} onChange={set("commission_pct")} />
        </div>
      </div>
    </div>
  );
}

function SummaryCard({ report }) {
  const rows = [
    { label: "Açılan/kapanan işlem", value: report.trades_closed, unit: "" },
    { label: "Kazanma oranı", value: fmt(report.win_rate_pct), unit: "%" },
    { label: "Toplam PnL", value: fmt(report.total_pnl_quote), unit: "$", pct: report.total_pnl_pct },
    { label: "Günlük ort. PnL", value: fmt(report.daily_pnl_quote), unit: "$", pct: report.daily_pnl_pct },
    { label: "Aylık ort. PnL", value: fmt(report.monthly_pnl_quote), unit: "$", pct: report.monthly_pnl_pct },
    { label: "Maks. düşüş (drawdown)", value: fmt(report.max_drawdown_pct), unit: "%" },
  ];
  return (
    <div className="card">
      <div className="card-title">Özet — {report.symbol} · {report.timeframe} · {report.candles_used.toLocaleString("tr-TR")} mum</div>
      <div className="muted">
        Dönem: {fmtDate(report.period_start)} → {fmtDate(report.period_end)} · Son bakiye: ${fmt(report.final_equity)}
        {report.created_at && <> · Çalıştırma: {fmtDate(report.created_at)}</>}
      </div>
      <div style={{ height: 8 }} />
      {rows.map((r) => (
        <div className="row" key={r.label}>
          <div className="row-value">{r.label}</div>
          <span className={"row-value " + (r.value >= 0 ? "pos" : "neg")}>
            {r.unit === "$" ? (r.value >= 0 ? "+" : "") : ""}
            {r.value}{r.unit}
            {r.pct !== undefined && ` (${r.pct >= 0 ? "+" : ""}${fmt(r.pct)}%)`}
          </span>
        </div>
      ))}
    </div>
  );
}

function WarningsCard({ warnings }) {
  return (
    <div className="card">
      <div className="card-title">Bilinen sınırlamalar</div>
      {warnings.map((w, i) => (
        <div key={i} className="muted" style={{ marginBottom: 6 }}>
          ⚠ {w}
        </div>
      ))}
    </div>
  );
}

function TradesCard({ trades }) {
  return (
    <div className="card">
      <div className="card-title">
        İşlem listesi
        <span className="pill">{trades.length}</span>
      </div>
      {!trades.length && <div className="muted">Bu parametrelerle hiç işlem kapanmadı.</div>}
      {trades.slice().reverse().map((t, i) => (
        <div className="row" key={i}>
          <div>
            <div className="row-value">
              {t.direction === "long" ? "Long" : "Short"} · ${fmt(t.entry_price)} → ${fmt(t.exit_price)}
            </div>
            <div className="muted">
              {fmtDate(t.entry_time)} → {fmtDate(t.exit_time)} · {t.duration_candles} mum ·{" "}
              {t.exit_reason === "stop_loss" ? "stop-loss" : "sinyal"}
            </div>
          </div>
          <span className={"row-value " + (t.pnl_pct >= 0 ? "pos" : "neg")}>
            {t.pnl_pct >= 0 ? "+" : ""}{fmt(t.pnl_pct)}%
          </span>
        </div>
      ))}
    </div>
  );
}

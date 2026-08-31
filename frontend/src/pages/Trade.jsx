import { useEffect, useState } from "react";
import { api } from "../api.js";
import ErrorBanner from "../components/ErrorBanner.jsx";

const TABS = ["DCA Optimizasyon", "Strateji Backtest", "Karar Motoru"];

export default function Trade() {
  const [tab, setTab] = useState(TABS[0]);

  return (
    <div className="page">
      <h1 className="page-title">Al-Sat</h1>
      <div className="tabs">
        {TABS.map((t) => (
          <button key={t} className={"tab-btn" + (tab === t ? " active" : "")} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </div>

      {tab === "DCA Optimizasyon" && <DcaOptimizer />}
      {tab === "Strateji Backtest" && <StrategyBacktest />}
      {tab === "Karar Motoru" && <EngineCycle />}
    </div>
  );
}

function DcaOptimizer() {
  const [form, setForm] = useState({ symbol: "BTC/USDT:USDT", balance: 500, direction: "long", objective: "profit_over_drawdown", top_n: 3 });
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const set = (key) => (e) => setForm({ ...form, [key]: e.target.value });

  const run = async () => {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await api.post("/dca/optimize", {
        symbol: form.symbol,
        balance: Number(form.balance),
        direction: form.direction,
        objective: form.objective,
        top_n: Number(form.top_n),
      });
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-title">En iyi DCA parametrelerini bul</div>
      <label className="field">Sembol</label>
      <input value={form.symbol} onChange={set("symbol")} placeholder="BTC/USDT:USDT" />
      <label className="field">Sermaye (USDT)</label>
      <input type="number" value={form.balance} onChange={set("balance")} />
      <label className="field">Yön</label>
      <select value={form.direction} onChange={set("direction")}>
        <option value="long">Long</option>
        <option value="short">Short</option>
      </select>
      <label className="field">Hedef</label>
      <select value={form.objective} onChange={set("objective")}>
        <option value="profit_over_drawdown">Getiri / Drawdown</option>
        <option value="profit">Toplam getiri</option>
        <option value="win_rate">Kazanma oranı</option>
      </select>

      <div style={{ height: 14 }} />
      <button className="primary" onClick={run} disabled={loading}>
        {loading ? "Hesaplanıyor..." : "Optimize Et"}
      </button>

      <ErrorBanner message={error} />

      {result?.candidates?.map((c, i) => (
        <div className="card" key={i} style={{ marginTop: 14, background: "var(--surface-2)" }}>
          <div className="card-title">#{i + 1} — Getiri %{c.total_profit_pct}</div>
          <div className="row"><span className="row-label">Deviation</span><span className="row-value">%{c.params.deviation_pct}</span></div>
          <div className="row"><span className="row-label">Safety orders</span><span className="row-value">{c.params.max_safety_orders}</span></div>
          <div className="row"><span className="row-label">Take profit</span><span className="row-value">%{c.params.take_profit_pct}</span></div>
          <div className="row"><span className="row-label">Kazanma oranı</span><span className="row-value">%{c.win_rate_pct}</span></div>
          <div className="row"><span className="row-label">Max drawdown</span><span className="row-value neg">%{c.max_drawdown_pct}</span></div>
          <div className="row"><span className="row-label">Kullanılan sermaye</span><span className="row-value">${c.max_capital_used}</span></div>
        </div>
      ))}
    </div>
  );
}

function StrategyBacktest() {
  const [examples, setExamples] = useState({});
  const [selected, setSelected] = useState("");
  const [symbol, setSymbol] = useState("BTC/USDT:USDT");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.get("/strategy/examples").then((data) => {
      setExamples(data);
      const first = Object.keys(data)[0];
      if (first) setSelected(first);
    }).catch((err) => setError(err.message));
  }, []);

  const run = async () => {
    if (!selected) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await api.post("/strategy/backtest", { symbol, strategy: examples[selected] });
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-title">Hazır stratejiyi test et</div>
      <label className="field">Strateji</label>
      <select value={selected} onChange={(e) => setSelected(e.target.value)}>
        {Object.entries(examples).map(([key, def]) => (
          <option key={key} value={key}>{def.name}</option>
        ))}
      </select>
      <label className="field">Sembol</label>
      <input value={symbol} onChange={(e) => setSymbol(e.target.value)} />

      <div style={{ height: 14 }} />
      <button className="primary" onClick={run} disabled={loading || !selected}>
        {loading ? "Test ediliyor..." : "Backtest Çalıştır"}
      </button>

      <ErrorBanner message={error} />

      {result && (
        <div className="card" style={{ marginTop: 14, background: "var(--surface-2)" }}>
          <div className="row"><span className="row-label">Kapanan işlem</span><span className="row-value">{result.trades_closed}</span></div>
          <div className="row"><span className="row-label">Kazanma oranı</span><span className="row-value">%{result.win_rate_pct}</span></div>
          <div className="row"><span className="row-label">Toplam getiri</span>
            <span className={"row-value " + (result.total_profit_pct >= 0 ? "pos" : "neg")}>%{result.total_profit_pct}</span>
          </div>
          <div className="row"><span className="row-label">Max drawdown</span><span className="row-value neg">%{result.max_drawdown_pct}</span></div>
        </div>
      )}
    </div>
  );
}

function EngineCycle() {
  const [actions, setActions] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true);
    setError("");
    setActions(null);
    try {
      const res = await api.post("/engine/run-cycle", {});
      setActions(res.actions);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-title">ML karar motorunu şimdi çalıştır</div>
      <p className="muted">
        Screener'ın Top Long/Short listesi üzerinde bir karar döngüsü çalıştırır (paper-trading).
        Model henüz eğitilmemişse hata döner — önce /ml/train ile eğitin.
      </p>
      <button className="primary" onClick={run} disabled={loading}>
        {loading ? "Çalışıyor..." : "Şimdi Çalıştır"}
      </button>

      <ErrorBanner message={error} />

      {actions?.length === 0 && <div className="muted" style={{ marginTop: 12 }}>Hiçbir aksiyon üretilmedi.</div>}
      {actions?.map((a, i) => (
        <div className="row" key={i}>
          <div>
            <div className="row-value">{a.symbol}</div>
            <div className="muted">{a.reason}</div>
          </div>
          <span className={"pill" + (a.type.startsWith("open") ? " active" : a.type === "blocked" ? " danger" : "")}>
            {a.type}
          </span>
        </div>
      ))}
    </div>
  );
}

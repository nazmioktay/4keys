import { useEffect, useState } from "react";
import { api } from "../api.js";
import Loading from "../components/Loading.jsx";
import ErrorBanner from "../components/ErrorBanner.jsx";

function fmt(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  return n.toLocaleString("tr-TR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

const KELLY_VARIANTS = [
  { key: "quarter", label: "Çeyrek Kelly", value: 0.25 },
  { key: "half", label: "Yarım Kelly", value: 0.5 },
  { key: "full", label: "Tam Kelly", value: 1.0 },
];

function weightsToText(weights) {
  return (weights || []).map((w) => Math.round(w * 100)).join(" / ");
}

function textToWeights(text) {
  const parts = text
    .split("/")
    .map((p) => Number(p.trim()))
    .filter((n) => !Number.isNaN(n) && n > 0);
  if (!parts.length) return null;
  const total = parts.reduce((a, b) => a + b, 0);
  return parts.map((p) => p / total);
}

export default function PaperTrading() {
  const [status, setStatus] = useState(null);
  const [pnl, setPnl] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [runningCycle, setRunningCycle] = useState(false);

  const load = async () => {
    setError("");
    try {
      const [s, p] = await Promise.all([api.get("/portfolio/status"), api.get("/portfolio/pnl")]);
      setStatus(s);
      setPnl(p);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const runCycle = async () => {
    setRunningCycle(true);
    setError("");
    try {
      await api.post("/engine/run-cycle", {});
      await load();
    } catch (err) {
      setError(err.message);
    } finally {
      setRunningCycle(false);
    }
  };

  if (loading) return <div className="page"><Loading /></div>;

  return (
    <div className="page">
      <h1 className="page-title">Paper Trading</h1>
      <ErrorBanner message={error} />

      <div className="card">
        <div className="card-title">
          Karar döngüsü
          <button className="primary" style={{ marginLeft: "auto", padding: "6px 14px" }} onClick={runCycle} disabled={runningCycle}>
            {runningCycle ? "Çalışıyor..." : "Şimdi Çalıştır"}
          </button>
        </div>
        <p className="muted">
          Zamanlayıcı zaten otomatik olarak periyodik çalıştırıyor (arka planda); burası anlık/manuel tetikleme içindir.
        </p>
      </div>

      <PnlCard pnl={pnl} />
      <OpenPositionsCard status={status} />
      <TrancheSettingsCard rules={status?.rules} onSaved={load} />
      <ClosedHistoryCard status={status} />
    </div>
  );
}

function PnlCard({ pnl }) {
  if (!pnl) return null;
  const rows = [
    { label: "Bugün (son 24s)", w: pnl.daily },
    { label: "Bu hafta (son 7g)", w: pnl.weekly },
    { label: "Bu ay (son 30g)", w: pnl.monthly },
    { label: "Toplam", w: pnl.total },
  ];
  return (
    <div className="card">
      <div className="card-title">PNL özeti</div>
      {rows.map((r) => (
        <div className="row" key={r.label}>
          <div>
            <div className="row-value">{r.label}</div>
            <div className="muted">{r.w.trade_count} işlem · %{fmt(r.w.win_rate_pct)} kazanma</div>
          </div>
          <span className={"row-value " + (r.w.pnl_quote >= 0 ? "pos" : "neg")}>
            {r.w.pnl_quote >= 0 ? "+" : ""}${fmt(r.w.pnl_quote)}
          </span>
        </div>
      ))}
    </div>
  );
}

function OpenPositionsCard({ status }) {
  return (
    <div className="card">
      <div className="card-title">
        Açık pozisyonlar (kademeli durum)
        <span className="pill">{status?.open_positions?.length ?? 0}</span>
      </div>
      {!status?.open_positions?.length && <div className="muted">Şu an açık pozisyon yok.</div>}
      {status?.open_positions?.map((p) => (
        <div className="row" key={p.symbol}>
          <div>
            <div className="row-value">{p.symbol}</div>
            <div className="muted">
              {p.direction === "long" ? "Long" : "Short"} · ${fmt(p.entry_price)} · alım {p.entry_fill_index}/{p.entry_tranche_count} dilim
              {p.exit_fill_index > 0 && ` · satış ${p.exit_fill_index}/${p.exit_tranche_count} dilim`}
            </div>
          </div>
          <div className="row-value">${fmt(p.size_quote)}</div>
        </div>
      ))}
    </div>
  );
}

function ClosedHistoryCard({ status }) {
  return (
    <div className="card">
      <div className="card-title">Son kapanan işlemler (dilimler dahil)</div>
      {!status?.closed_history?.length && <div className="muted">Henüz kapanan işlem yok.</div>}
      {status?.closed_history?.slice().reverse().slice(0, 10).map((t, i) => (
        <div className="row" key={i}>
          <div>
            <div className="row-value">
              {t.symbol}
              {t.tranche ? <span className="muted"> · dilim {t.tranche}</span> : null}
            </div>
            <div className="muted">${fmt(t.size_quote)}{t.partial ? " · kısmi" : ""}</div>
          </div>
          <span className={"row-value " + (t.pnl_pct >= 0 ? "pos" : "neg")}>
            {t.pnl_pct >= 0 ? "+" : ""}{fmt(t.pnl_pct)}%
          </span>
        </div>
      ))}
    </div>
  );
}

function TrancheSettingsCard({ rules, onSaved }) {
  const [form, setForm] = useState(null);
  const [entryText, setEntryText] = useState("");
  const [exitText, setExitText] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [savedMsg, setSavedMsg] = useState("");

  useEffect(() => {
    if (rules && !form) {
      setForm(rules);
      setEntryText(weightsToText(rules.entry_tranche_weights));
      setExitText(weightsToText(rules.exit_tranche_weights));
    }
  }, [rules]);

  if (!form) return null;

  const applyKellyVariant = (value) => setForm({ ...form, kelly_multiplier: value });

  const save = async () => {
    setError("");
    setSavedMsg("");
    const entryWeights = textToWeights(entryText);
    const exitWeights = textToWeights(exitText);
    if (!entryWeights || !exitWeights) {
      setError("Dilim ağırlıkları geçersiz — örn. '50 / 50' formatında girin.");
      return;
    }
    setSaving(true);
    try {
      const payload = { ...form, entry_tranche_weights: entryWeights, exit_tranche_weights: exitWeights };
      const res = await api.put("/portfolio/rules", payload);
      setForm(res);
      setSavedMsg("Kaydedildi.");
      onSaved();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="card">
      <div className="card-title">Parametrik ayarlar</div>

      <label className="field">Kelly boyutlandırma yöntemi</label>
      <select
        value={form.position_sizing_method}
        onChange={(e) => setForm({ ...form, position_sizing_method: e.target.value })}
      >
        <option value="fixed_risk">Sabit risk (SL mesafesine göre)</option>
        <option value="kelly">Kelly kriteri</option>
      </select>

      {form.position_sizing_method === "kelly" && (
        <>
          <label className="field">Kelly çeşidi</label>
          <div className="tabs">
            {KELLY_VARIANTS.map((v) => (
              <button
                key={v.key}
                className={"tab-btn" + (Math.abs(form.kelly_multiplier - v.value) < 0.001 ? " active" : "")}
                onClick={() => applyKellyVariant(v.value)}
              >
                {v.label}
              </button>
            ))}
          </div>
        </>
      )}

      <label className="field">Kademeli alım dilimleri (%, "50 / 50" gibi)</label>
      <input value={entryText} onChange={(e) => setEntryText(e.target.value)} placeholder="50 / 50" />
      <div className="muted">İlk dilim hemen açılır; sonraki dilim(ler) sinyal bir sonraki döngüde de kalıcıysa eklenir.</div>

      <label className="field">Kademeli satış dilimleri (%, "50 / 50" gibi)</label>
      <input value={exitText} onChange={(e) => setExitText(e.target.value)} placeholder="50 / 50" />
      <div className="muted">Kapanış sinyali geldiğinde ilk dilim satılır; kalan sinyal sonraki döngüde de sürerse tamamı kapanır.</div>

      <div style={{ height: 14 }} />
      <button className="primary" onClick={save} disabled={saving}>
        {saving ? "Kaydediliyor..." : "Kaydet"}
      </button>
      {savedMsg && <div className="pos" style={{ marginTop: 8 }}>{savedMsg}</div>}
      <ErrorBanner message={error} />
    </div>
  );
}

import { useEffect, useState } from "react";
import { api } from "../api.js";
import ErrorBanner from "../components/ErrorBanner.jsx";
import Loading from "../components/Loading.jsx";

export default function Settings() {
  const [rules, setRules] = useState(null);
  const [security, setSecurity] = useState(null);
  const [scheduler, setScheduler] = useState(null);
  const [db, setDb] = useState(null);
  const [error, setError] = useState("");
  const [saveMsg, setSaveMsg] = useState("");
  const [loading, setLoading] = useState(true);

  const loadAll = async () => {
    setError("");
    try {
      const [p, s, sch, dbStatus] = await Promise.all([
        api.get("/portfolio/status"),
        api.get("/security/status"),
        api.get("/scheduler/status"),
        api.get("/db/status"),
      ]);
      setRules(p.rules);
      setSecurity(s);
      setScheduler(sch);
      setDb(dbStatus);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAll();
  }, []);

  const saveRules = async () => {
    setSaveMsg("");
    setError("");
    try {
      await api.put("/portfolio/rules", rules);
      setSaveMsg("Kaydedildi.");
    } catch (err) {
      setError(err.message);
    }
  };

  const toggleKillSwitch = async () => {
    setError("");
    try {
      if (security.kill_switch.active) {
        await api.post("/security/kill-switch/deactivate", {});
      } else {
        await api.post("/security/kill-switch/activate", { reason: "Manuel olarak arayüzden durduruldu" });
      }
      await loadAll();
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading) return <div className="page"><Loading /></div>;

  const setRule = (key) => (e) => {
    const value = e.target.type === "checkbox" ? e.target.checked : Number(e.target.value);
    setRules({ ...rules, [key]: value });
  };

  return (
    <div className="page">
      <h1 className="page-title">Ayarlar</h1>
      <ErrorBanner message={error} />

      <div className="card">
        <div className="card-title">Güvenlik</div>
        <div className="row">
          <span className="row-label">Kill switch</span>
          <span className={"pill" + (security?.kill_switch?.active ? " danger" : " active")}>
            {security?.kill_switch?.active ? "AKTİF" : "kapalı"}
          </span>
        </div>
        {security?.kill_switch?.active && <p className="muted">Sebep: {security.kill_switch.reason}</p>}
        <div className="row">
          <span className="row-label">Canlı işlem (Binance)</span>
          <span className="row-value">{security?.live_trading_enabled ? "açık" : "kapalı"}</span>
        </div>
        <div className="row">
          <span className="row-label">Maksimum kaldıraç</span>
          <span className="row-value">{security?.max_leverage}x</span>
        </div>
        <div style={{ height: 10 }} />
        <button className={security?.kill_switch?.active ? "secondary" : "danger secondary"} onClick={toggleKillSwitch}>
          {security?.kill_switch?.active ? "Kill Switch'i Kapat" : "Kill Switch'i Aktive Et"}
        </button>
      </div>

      <div className="card">
        <div className="card-title">Pozisyon boyutlandırma kuralları</div>
        <label className="field">Yöntem</label>
        <select
          value={rules.position_sizing_method}
          onChange={(e) => setRules({ ...rules, position_sizing_method: e.target.value })}
        >
          <option value="fixed_risk">Sabit risk</option>
          <option value="kelly">Kelly kriteri</option>
        </select>

        <label className="field">İşlem başına risk (%)</label>
        <input type="number" step="0.1" value={rules.max_risk_per_trade_pct} onChange={setRule("max_risk_per_trade_pct")} />

        <label className="field">Kelly çarpanı (0.25=çeyrek, 0.5=yarım, 1=tam)</label>
        <input type="number" step="0.05" value={rules.kelly_multiplier} onChange={setRule("kelly_multiplier")} />

        <label className="field">Maksimum toplam maruziyet (%)</label>
        <input type="number" value={rules.max_total_exposure_pct} onChange={setRule("max_total_exposure_pct")} />

        <label className="field">Sembol başına maksimum maruziyet (%)</label>
        <input type="number" value={rules.max_symbol_exposure_pct} onChange={setRule("max_symbol_exposure_pct")} />

        <label className="field">Maksimum eşzamanlı pozisyon</label>
        <input type="number" value={rules.max_concurrent_positions} onChange={setRule("max_concurrent_positions")} />

        <label className="field">Günlük zarar limiti (%)</label>
        <input type="number" value={rules.daily_loss_limit_pct} onChange={setRule("daily_loss_limit_pct")} />

        <div style={{ height: 14 }} />
        <button className="primary" onClick={saveRules}>Kaydet</button>
        {saveMsg && <p className="muted" style={{ marginTop: 8 }}>{saveMsg}</p>}
      </div>

      <div className="card">
        <div className="card-title">Zamanlayıcı</div>
        <div className="row">
          <span className="row-label">Durum</span>
          <span className="row-value">{scheduler?.running ? "çalışıyor" : "durdu"}</span>
        </div>
        {scheduler?.jobs?.map((j) => (
          <div className="row" key={j.job_id}>
            <span className="row-label">{j.job_id}</span>
            <span className="row-value">{j.last_run_at ? (j.ok ? "✓" : "✗") : "henüz çalışmadı"}</span>
          </div>
        ))}
      </div>

      <div className="card">
        <div className="card-title">Veritabanı</div>
        <div className="row">
          <span className="row-label">Etkin</span>
          <span className="row-value">{db?.enabled ? "evet" : "hayır (bellek içi)"}</span>
        </div>
        {db?.enabled && (
          <div className="row">
            <span className="row-label">Bağlantı</span>
            <span className="row-value">{db.connected ? "başarılı" : "başarısız"}</span>
          </div>
        )}
      </div>
    </div>
  );
}

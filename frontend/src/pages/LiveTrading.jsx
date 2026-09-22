import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import Loading from "../components/Loading.jsx";
import ErrorBanner from "../components/ErrorBanner.jsx";

function fmt(n, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  return Number(n).toLocaleString("tr-TR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export default function LiveTrading() {
  const [security, setSecurity] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const loadSecurity = async () => {
    try {
      setSecurity(await api.get("/security/status"));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSecurity();
  }, []);

  if (loading) return <div className="page"><Loading /></div>;

  const gatesOpen = security?.live_trading_enabled && !security?.kill_switch?.active;

  return (
    <div className="page">
      <h1 className="page-title">Canlı İşlem</h1>
      <p className="muted">
        <Link to="/settings">← Ayarlar'a dön</Link>
      </p>
      <ErrorBanner message={error} />

      <div className={"card"} style={{ borderColor: gatesOpen ? "var(--danger, #e5484d)" : undefined }}>
        <div className="card-title">⚠ Gerçek para uyarısı</div>
        <p className="muted">
          Bu ekrandaki emirler `.env` dosyasındaki Binance anahtarlarına bağlı hesapta çalışır.
          {security?.binance_testnet
            ? " Şu an testnet modunda görünüyor, ANCAK bu proje için gerçek hesap anahtarları kullanıldığı bildirildi — dikkatli olun."
            : " Testnet KAPALI — bu GERÇEK hesap ve GERÇEK para demektir."}
        </p>
        <div className="row">
          <span className="row-label">Canlı işlem anahtarı (FOURKEYS_ENABLE_LIVE_TRADING)</span>
          <span className={"pill" + (security?.live_trading_enabled ? " danger" : "")}>
            {security?.live_trading_enabled ? "AÇIK" : "kapalı"}
          </span>
        </div>
        <div className="row">
          <span className="row-label">Kill switch</span>
          <span className={"pill" + (security?.kill_switch?.active ? " danger" : " active")}>
            {security?.kill_switch?.active ? "AKTİF (emir engellenir)" : "kapalı"}
          </span>
        </div>
        {!security?.live_trading_enabled && (
          <p className="muted" style={{ marginTop: 8 }}>
            Emir göndermeden önce backend'de <code>FOURKEYS_ENABLE_LIVE_TRADING=true</code> yapıp yeniden başlatmanız gerekir.
            Bu anahtar bilinçli olarak arayüzden değil, yalnızca `.env`'den açılabilir.
          </p>
        )}
      </div>

      <BalanceCard />
      <PositionsCard gatesOpen={gatesOpen} />
      <OrderForm gatesOpen={gatesOpen} maxLeverage={security?.max_leverage} onOrderPlaced={loadSecurity} />
    </div>
  );
}

function BalanceCard() {
  const [marketType, setMarketType] = useState("future");
  const [balance, setBalance] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      setBalance(await api.get(`/trading/balance?market_type=${marketType}`));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [marketType]);

  const usdt = balance?.total?.USDT ?? balance?.USDT?.total;

  return (
    <div className="card">
      <div className="card-title">
        Bakiye
        <select
          style={{ marginLeft: "auto", width: "auto" }}
          value={marketType}
          onChange={(e) => setMarketType(e.target.value)}
        >
          <option value="future">Futures</option>
          <option value="spot">Spot</option>
        </select>
      </div>
      <button className="secondary" onClick={load} disabled={loading}>
        {loading ? "Sorgulanıyor..." : "Yenile"}
      </button>
      <ErrorBanner message={error} />
      {balance && (
        <div className="row" style={{ marginTop: 10 }}>
          <span className="row-label">USDT</span>
          <span className="row-value">{usdt !== undefined ? fmt(usdt, 2) : "bulunamadı (ham yanıtı kontrol edin)"}</span>
        </div>
      )}
    </div>
  );
}

function PositionsCard({ gatesOpen }) {
  const [positions, setPositions] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [armedSymbol, setArmedSymbol] = useState(null);
  const [closingSymbol, setClosingSymbol] = useState(null);
  const [closeError, setCloseError] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.get("/trading/positions");
      setPositions((res || []).filter((p) => Number(p.contracts ?? p.positionAmt ?? 0) !== 0));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const armClose = (symbol) => {
    setArmedSymbol(symbol);
    setTimeout(() => setArmedSymbol((cur) => (cur === symbol ? null : cur)), 5000);
  };

  const closePosition = async (p) => {
    setCloseError("");
    setClosingSymbol(p.symbol);
    try {
      const closeSide = p.side === "long" ? "sell" : "buy";
      const amount = Math.abs(Number(p.contracts ?? p.positionAmt));
      await api.post("/trading/order", {
        symbol: p.symbol,
        side: closeSide,
        order_type: "market",
        amount,
        market_type: "future",
        reduce_only: true,
        confirm: true,
      });
      setArmedSymbol(null);
      await load();
    } catch (err) {
      setCloseError(err.message);
    } finally {
      setClosingSymbol(null);
    }
  };

  return (
    <div className="card">
      <div className="card-title">
        Açık pozisyonlar (gerçek hesap)
        <button className="secondary" style={{ marginLeft: "auto", width: "auto", padding: "4px 12px" }} onClick={load} disabled={loading}>
          ↻
        </button>
      </div>
      <ErrorBanner message={error} />
      <ErrorBanner message={closeError} />
      {positions && positions.length === 0 && <div className="muted" style={{ marginTop: 8 }}>Açık pozisyon yok.</div>}
      {positions?.map((p, i) => (
        <div className="row" key={i}>
          <div>
            <div className="row-value">{p.symbol}</div>
            <div className="muted">
              {p.side} · giriş ${fmt(p.entryPrice, 2)} · {fmt(p.contracts ?? p.positionAmt)}
            </div>
          </div>
          <button
            className={"secondary" + (armedSymbol === p.symbol ? " danger" : "")}
            style={{ width: "auto", padding: "6px 12px" }}
            disabled={!gatesOpen || closingSymbol === p.symbol}
            onClick={() => (armedSymbol === p.symbol ? closePosition(p) : armClose(p.symbol))}
          >
            {closingSymbol === p.symbol ? "Kapatılıyor..." : armedSymbol === p.symbol ? "Emin misin? Kapat" : "Kapat"}
          </button>
        </div>
      ))}
    </div>
  );
}

const LEVERAGE_OPTIONS = [1, 2, 3];
const MAINTENANCE_MARGIN_RATE = 0.004; // yaklaşık, düşük notional/BTC-USDT alt dilimi varsayımı

function estimateLiquidationPrice(entryPrice, leverage, direction) {
  if (!entryPrice || !leverage) return null;
  const factor = 1 / leverage - MAINTENANCE_MARGIN_RATE;
  return direction === "long" ? entryPrice * (1 - factor) : entryPrice * (1 + factor);
}

function OrderForm({ gatesOpen, maxLeverage, onOrderPlaced }) {
  const [symbol, setSymbol] = useState("BTC/USDT:USDT");
  const [marketType, setMarketType] = useState("future");
  const [direction, setDirection] = useState("long");
  const [leverage, setLeverage] = useState(1);
  const [orderType, setOrderType] = useState("market");
  const [usdtAmount, setUsdtAmount] = useState("100");
  const [limitPrice, setLimitPrice] = useState("");
  const [confirmChecked, setConfirmChecked] = useState(false);

  const [price, setPrice] = useState(null);
  const [priceError, setPriceError] = useState("");
  const [priceLoading, setPriceLoading] = useState(false);

  const [budget, setBudget] = useState(null);
  const [budgetError, setBudgetError] = useState("");

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [sending, setSending] = useState(false);

  const isFuture = marketType === "future";
  const cap = Math.min(maxLeverage || 3, 3);

  const loadPrice = async () => {
    setPriceLoading(true);
    setPriceError("");
    try {
      const res = await api.get(`/trading/price?symbol=${encodeURIComponent(symbol)}&market_type=${marketType}`);
      setPrice(res.last);
    } catch (err) {
      setPrice(null);
      setPriceError(err.message);
    } finally {
      setPriceLoading(false);
    }
  };

  const loadBudget = async () => {
    setBudgetError("");
    try {
      const res = await api.get(`/trading/balance?market_type=${marketType}`);
      const usdt = res?.total?.USDT ?? res?.USDT?.total ?? null;
      setBudget(usdt);
    } catch (err) {
      setBudget(null);
      setBudgetError(err.message);
    }
  };

  useEffect(() => {
    loadPrice();
    loadBudget();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, marketType]);

  const effectivePrice = orderType === "limit" && Number(limitPrice) > 0 ? Number(limitPrice) : price;
  const baseAmount = effectivePrice ? Number(usdtAmount) / effectivePrice : null;
  const liquidationPrice = isFuture ? estimateLiquidationPrice(effectivePrice, leverage, direction) : null;

  const submit = async () => {
    setError("");
    setResult(null);
    setSending(true);
    try {
      if (isFuture) {
        await api.post("/trading/leverage", { symbol, leverage, confirm: true });
      }
      const payload = {
        symbol,
        side: direction === "long" ? "buy" : "sell",
        order_type: orderType,
        amount: baseAmount,
        market_type: marketType,
        confirm: true,
      };
      if (orderType === "limit") payload.price = Number(limitPrice);
      const res = await api.post("/trading/order", payload);
      setResult(res);
      setConfirmChecked(false);
      onOrderPlaced?.();
      loadBudget();
    } catch (err) {
      setError(err.message);
    } finally {
      setSending(false);
    }
  };

  const canSubmit = gatesOpen && confirmChecked && baseAmount > 0 && !sending;

  return (
    <div className="card">
      <div className="card-title">Emir gönder</div>

      <label className="field">Sembol</label>
      <input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="BTC/USDT:USDT" />

      <label className="field">Piyasa</label>
      <select value={marketType} onChange={(e) => setMarketType(e.target.value)}>
        <option value="future">Futures</option>
        <option value="spot">Spot</option>
      </select>

      <div className="row" style={{ marginTop: 12 }}>
        <span className="row-label">Güncel fiyat</span>
        <span className="row-value">
          {priceLoading ? "..." : price !== null ? `$${fmt(price, 2)}` : "-"}
          {" "}
          <button className="secondary" style={{ width: "auto", padding: "2px 10px", display: "inline" }} onClick={loadPrice} disabled={priceLoading}>
            ↻
          </button>
        </span>
      </div>
      <ErrorBanner message={priceError} />

      <div className="row">
        <span className="row-label">Toplam bütçe (USDT)</span>
        <span className="row-value">{budget !== null ? `$${fmt(budget, 2)}` : "-"}</span>
      </div>
      <ErrorBanner message={budgetError} />

      <label className="field">Yön</label>
      <div className="tabs">
        <button className={"tab-btn" + (direction === "long" ? " active" : "")} onClick={() => setDirection("long")}>
          Long (al)
        </button>
        <button className={"tab-btn" + (direction === "short" ? " active" : "")} onClick={() => setDirection("short")}>
          Short (sat)
        </button>
      </div>

      {isFuture && (
        <>
          <label className="field">Kaldıraç</label>
          <div className="tabs">
            {LEVERAGE_OPTIONS.filter((l) => l <= cap).map((l) => (
              <button key={l} className={"tab-btn" + (leverage === l ? " active" : "")} onClick={() => setLeverage(l)}>
                {l}x
              </button>
            ))}
          </div>
          <p className="muted">Kod içi güvenlik tavanı: {cap}x — bu değer arayüzden aşılamaz.</p>
        </>
      )}

      <label className="field">Emir tipi</label>
      <select value={orderType} onChange={(e) => setOrderType(e.target.value)}>
        <option value="market">Market</option>
        <option value="limit">Limit</option>
      </select>

      {orderType === "limit" && (
        <>
          <label className="field">Limit fiyatı</label>
          <input type="number" step="0.01" value={limitPrice} onChange={(e) => setLimitPrice(e.target.value)} />
        </>
      )}

      <label className="field">Miktar (USDT)</label>
      <input type="number" step="1" value={usdtAmount} onChange={(e) => setUsdtAmount(e.target.value)} />
      <div className="muted">
        ≈ {baseAmount ? fmt(baseAmount, 6) : "-"} {symbol.split("/")[0]}
      </div>

      {isFuture && (
        <div className="row" style={{ marginTop: 10 }}>
          <span className="row-label">Tahmini likidasyon fiyatı</span>
          <span className="row-value neg">{liquidationPrice ? `$${fmt(liquidationPrice, 2)}` : "-"}</span>
        </div>
      )}
      {isFuture && liquidationPrice && (
        <p className="muted">
          Yaklaşık değerdir (izole marjin, sabit %{(MAINTENANCE_MARGIN_RATE * 100).toFixed(2)} sürdürme marjı varsayımıyla) — Binance'in gerçek
          hesaplaması pozisyon büyüklüğüne göre kademeli değişir.
        </p>
      )}

      <div style={{ height: 14 }} />
      <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
        <input type="checkbox" checked={confirmChecked} onChange={(e) => setConfirmChecked(e.target.checked)} />
        <span>Onaylıyorum: bu emir gerçek hesapta çalışır ve gerçek para kullanır.</span>
      </label>

      <div style={{ height: 14 }} />
      <button className="danger primary" onClick={submit} disabled={!canSubmit}>
        {sending ? "Gönderiliyor..." : "Emri Gönder"}
      </button>
      {!gatesOpen && (
        <p className="muted" style={{ marginTop: 8 }}>
          Canlı işlem anahtarı kapalı olduğu için buton devre dışı — üstteki uyarı kartına bakın.
        </p>
      )}

      <ErrorBanner message={error} />
      {result && (
        <div className="card" style={{ marginTop: 14, background: "var(--surface-2)" }}>
          <div className="card-title">Emir sonucu</div>
          <pre style={{ whiteSpace: "pre-wrap", fontSize: 12, margin: 0 }}>{JSON.stringify(result.raw, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

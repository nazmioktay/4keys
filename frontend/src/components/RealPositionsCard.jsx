import { useEffect, useState } from "react";
import { api } from "../api.js";
import ErrorBanner from "./ErrorBanner.jsx";

function fmt(n, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  return Number(n).toLocaleString("tr-TR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

// Binance'teki GERÇEK hesabın açık pozisyonlarını gösterir + kapatma imkanı
// sunar — hem Portföy (özet/genel bakış) hem Canlı İşlem (emir formunun
// yanında) sayfasında aynı davranışla kullanılır.
export default function RealPositionsCard({ gatesOpen }) {
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
      {positions?.map((p, i) => {
        const pnl = Number(p.unrealizedPnl ?? p.info?.unRealizedProfit ?? 0);
        return (
          <div className="row" key={i}>
            <div>
              <div className="row-value">{p.symbol}</div>
              <div className="muted">
                {p.side} · giriş ${fmt(p.entryPrice, 2)} · {fmt(p.contracts ?? p.positionAmt)}
                {" · "}
                <span className={pnl >= 0 ? "pos" : "neg"}>
                  {pnl >= 0 ? "+" : ""}${fmt(pnl, 2)}
                </span>
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
        );
      })}
    </div>
  );
}

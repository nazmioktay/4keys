import { NavLink, Route, Routes } from "react-router-dom";
import Portfolio from "./pages/Portfolio.jsx";
import Trade from "./pages/Trade.jsx";
import PaperTrading from "./pages/PaperTrading.jsx";
import Screener from "./pages/Screener.jsx";
import Assistant from "./pages/Assistant.jsx";
import Settings from "./pages/Settings.jsx";
import Backtest from "./pages/Backtest.jsx";

const TABS = [
  { to: "/", label: "Portföy", icon: "◔", end: true },
  { to: "/trade", label: "Al-Sat", icon: "⇅" },
  { to: "/paper-trading", label: "Paper Trading", icon: "◫" },
  { to: "/backtest", label: "Backtest", icon: "⏱" },
  { to: "/assistant", label: "AI Asistan", icon: "✦" },
  { to: "/screener", label: "Araştırıcı", icon: "▤" },
  { to: "/settings", label: "Ayarlar", icon: "⚙" },
];

// Grafana, 4keys'in kendi frontend'inden ayrı bir izleme aracı olarak
// çalışıyor (bkz. README "Grafana/Prometheus") — burada yeniden inşa
// edilmiyor, aynı host üzerindeki :3001 portuna yeni sekmede yönlendiriliyor.
const GRAFANA_URL = `${window.location.protocol}//${window.location.hostname}:3001`;

export default function App() {
  return (
    <div className="app-shell">
      <Routes>
        <Route path="/" element={<Portfolio />} />
        <Route path="/trade" element={<Trade />} />
        <Route path="/paper-trading" element={<PaperTrading />} />
        <Route path="/backtest" element={<Backtest />} />
        <Route path="/assistant" element={<Assistant />} />
        <Route path="/screener" element={<Screener />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>

      <nav className="bottom-nav">
        {TABS.map((tab) => (
          <NavLink
            key={tab.to}
            to={tab.to}
            end={tab.end}
            className={({ isActive }) => "nav-item" + (isActive ? " active" : "")}
          >
            <span className="nav-icon">{tab.icon}</span>
            <span>{tab.label}</span>
          </NavLink>
        ))}
        <a href={GRAFANA_URL} target="_blank" rel="noopener noreferrer" className="nav-item">
          <span className="nav-icon">📈</span>
          <span>Monitoring</span>
        </a>
      </nav>
    </div>
  );
}

import { NavLink, Route, Routes } from "react-router-dom";
import Portfolio from "./pages/Portfolio.jsx";
import PaperTrading from "./pages/PaperTrading.jsx";
import Otopilot from "./pages/Otopilot.jsx";
import Settings from "./pages/Settings.jsx";
import Backtest from "./pages/Backtest.jsx";
import LiveTrading from "./pages/LiveTrading.jsx";

const TABS = [
  { to: "/", label: "Portföy", icon: "◔", end: true },
  { to: "/backtest", label: "Backtest", icon: "⏱" },
  { to: "/paper-trading", label: "Paper Trading", icon: "◫" },
  { to: "/otopilot", label: "Otopilot", icon: "◉" },
  { to: "/live-trading", label: "Canlı İşlem", icon: "⚠" },
  { to: "/settings", label: "Ayarlar", icon: "⚙" },
];

export default function App() {
  return (
    <div className="app-shell">
      <Routes>
        <Route path="/" element={<Portfolio />} />
        <Route path="/paper-trading" element={<PaperTrading />} />
        <Route path="/otopilot" element={<Otopilot />} />
        <Route path="/backtest" element={<Backtest />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/live-trading" element={<LiveTrading />} />
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
      </nav>
    </div>
  );
}

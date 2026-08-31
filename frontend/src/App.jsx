import { NavLink, Route, Routes } from "react-router-dom";
import Portfolio from "./pages/Portfolio.jsx";
import Trade from "./pages/Trade.jsx";
import Screener from "./pages/Screener.jsx";
import Assistant from "./pages/Assistant.jsx";
import Settings from "./pages/Settings.jsx";

const TABS = [
  { to: "/", label: "Portföy", icon: "◔", end: true },
  { to: "/trade", label: "Al-Sat", icon: "⇅" },
  { to: "/assistant", label: "AI Asistan", icon: "✦" },
  { to: "/screener", label: "Araştırıcı", icon: "▤" },
  { to: "/settings", label: "Ayarlar", icon: "⚙" },
];

export default function App() {
  return (
    <div className="app-shell">
      <Routes>
        <Route path="/" element={<Portfolio />} />
        <Route path="/trade" element={<Trade />} />
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
      </nav>
    </div>
  );
}

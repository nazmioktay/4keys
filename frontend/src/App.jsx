import { Navigate, NavLink, Route, Routes, useLocation } from "react-router-dom";
import Portfolio from "./pages/Portfolio.jsx";
import PaperTrading from "./pages/PaperTrading.jsx";
import Otopilot from "./pages/Otopilot.jsx";
import Settings from "./pages/Settings.jsx";
import Backtest from "./pages/Backtest.jsx";
import LiveTrading from "./pages/LiveTrading.jsx";
import Login from "./pages/Login.jsx";
import { auth } from "./api.js";

const TABS = [
  { to: "/", label: "Portföy", icon: "◔", end: true },
  { to: "/backtest", label: "Backtest", icon: "⏱" },
  { to: "/paper-trading", label: "Paper Trading", icon: "◫" },
  { to: "/otopilot", label: "Otopilot", icon: "◉" },
  { to: "/live-trading", label: "Canlı İşlem", icon: "⚠" },
  { to: "/settings", label: "Ayarlar", icon: "⚙" },
];

function RequireAuth({ children }) {
  const location = useLocation();
  if (!auth.getToken()) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }
  return children;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        path="*"
        element={
          <RequireAuth>
            <AppShell />
          </RequireAuth>
        }
      />
    </Routes>
  );
}

function AppShell() {
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

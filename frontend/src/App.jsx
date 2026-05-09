import React, { useEffect, useState } from "react";
import { apiFetch } from "./utils/api";
import Sidebar from "./components/Sidebar";
import Toast from "./components/Toast";
import Landing from "./pages/Landing";
import AuthPage from "./pages/AuthPage";
import AdminLogin from "./pages/AdminLogin";
import Dashboard from "./pages/Dashboard";
import AnalysisPage from "./pages/AnalysisPage";
import HistoryPage from "./pages/HistoryPage";
import AdminDashboard from "./pages/AdminDashboard";

export default function App() {
  const storedUser = JSON.parse(localStorage.getItem("ra_user") || "null");
  const storedToken = localStorage.getItem("ra_token") || null;
  const [page, setPage] = useState(() => (storedUser ? (storedUser.is_admin ? "admin" : "dashboard") : "landing"));
  const [user, setUser] = useState(storedUser);
  const [token, setToken] = useState(storedToken);
  const [analyses, setAnalyses] = useState([]);
  const [toast, setToast] = useState(null);

  useEffect(() => {
    if (!user || !token) return;
    if (page === "landing" || page === "login" || page === "signup" || page === "admin-login") {
      setPage(user.is_admin ? "admin" : "dashboard");
    }
  }, [page, user, token]);

  const nav = (p) => setPage(p);
  const notify = (msg, type = "info") => setToast({ msg, type });
  const loadAnalyses = async () => {
    if (!token || !user || user.is_admin) return setAnalyses([]);
    try { setAnalyses(await apiFetch("/api/analyses")); } catch (e) { notify(e.message, "error"); }
  };

  useEffect(() => { loadAnalyses(); }, [token, user]);

  const onLogin = (u, jwt) => {
    const t = jwt || localStorage.getItem("ra_token") || null;
    if (jwt) localStorage.setItem("ra_token", jwt);
    setUser(u); setToken(t); localStorage.setItem("ra_user", JSON.stringify(u));
    setPage(u.is_admin ? "admin" : "dashboard");
  };

  const onLogout = () => {
    setUser(null); setToken(null); setAnalyses([]);
    localStorage.removeItem("ra_token"); localStorage.removeItem("ra_user");
    setPage("landing");
  };

  const shell = (content) => (<><div style={{ display: "flex", minHeight: "100vh" }}><Sidebar page={page} nav={nav} user={user} onLogout={onLogout} /><main style={{ marginLeft: 240, flex: 1 }}>{content}</main></div>{toast && <Toast {...toast} onClose={() => setToast(null)} />}</>);

  if (page === "landing") return <><Landing nav={nav} />{toast && <Toast {...toast} onClose={() => setToast(null)} />}</>;
  if (page === "login" || page === "signup") return <><AuthPage mode={page} nav={nav} onLogin={onLogin} />{toast && <Toast {...toast} onClose={() => setToast(null)} />}</>;
  if (page === "admin-login") return <><AdminLogin nav={nav} onAdminLogin={onLogin} />{toast && <Toast {...toast} onClose={() => setToast(null)} />}</>;
  if (page === "admin") return <><AdminDashboard nav={nav} onLogout={onLogout} toast={notify} />{toast && <Toast {...toast} onClose={() => setToast(null)} />}</>;
  if (!user) return <><Landing nav={nav} />{toast && <Toast {...toast} onClose={() => setToast(null)} />}</>;
  if (page === "analysis") return shell(<AnalysisPage user={user} onNewAnalysis={(a) => setAnalyses((prev) => [a, ...prev])} toast={notify} />);
  if (page === "history") return shell(<HistoryPage analyses={analyses} onRefresh={loadAnalyses} toast={notify} user={user} />);
  return shell(<Dashboard user={user} nav={nav} analyses={analyses} />);
}

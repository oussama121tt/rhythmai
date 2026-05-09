import { useEffect, useState } from "react";

export default function NotificationBell({ apiFetch }) {
  const [notifs, setNotifs] = useState([]);
  const [open, setOpen] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);
  const [lastUpdated, setLastUpdated] = useState(null);

  const loadNotifications = async () => {
    try {
      const data = await apiFetch("/api/notifications");
      setNotifs(data.notifications || []);
      setUnreadCount(data.unread_count || 0);
      setLastUpdated(data.last_updated || null);
    } catch {
      setNotifs([]);
      setUnreadCount(0);
      setLastUpdated(null);
    }
  };

  useEffect(() => {
    loadNotifications();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      apiFetch("/api/notifications")
        .then((data) => {
          setNotifs(data.notifications || []);
          setUnreadCount(data.unread_count || 0);
          setLastUpdated(data.last_updated || null);
        })
        .catch(() => {});
    }, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [apiFetch]);

  const handleOpen = () => {
    const nextOpen = !open;
    setOpen(nextOpen);
    if (nextOpen && unreadCount > 0) {
      setUnreadCount(0);
      apiFetch("/api/notifications/mark-read", { method: "POST" }).catch(() => {});
    }
  };

  return (
    <div style={{ position: "relative", display: "inline-block" }}>
      <button
        onClick={handleOpen}
        style={{ background: "none", border: "none", cursor: "pointer", fontSize: 22, position: "relative" }}
        title="Nouvelles publications ECG"
        type="button"
      >
        🔔
        {unreadCount > 0 && (
          <span style={{
            position: "absolute",
            top: -4,
            right: -4,
            background: "#e53e3e",
            color: "#fff",
            borderRadius: "50%",
            fontSize: 11,
            width: 18,
            height: 18,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}>{unreadCount}</span>
        )}
      </button>

      {open && (
        <div style={{
          position: "absolute",
          right: 0,
          top: 36,
          width: 360,
          maxHeight: 420,
          overflowY: "auto",
          background: "#fff",
          border: "1px solid #e2e8f0",
          borderRadius: 10,
          boxShadow: "0 8px 24px rgba(0,0,0,0.12)",
          zIndex: 1000,
          padding: 12,
        }}>
          <p style={{ fontWeight: 700, marginBottom: 8, color: "#2d3748" }}>
            📄 Nouvelles publications ECG
          </p>
          <span style={{fontSize:11, color:"rgba(255,255,255,0.8)"}}>
            Mis à jour: {lastUpdated ? new Date(lastUpdated).toLocaleDateString('fr-FR') : '...'}
          </span>
          {notifs.length === 0 && <p style={{ color: "#718096" }}>Aucune nouvelle publication.</p>}
          {notifs.map((n) => (
            <div key={n.pmid} style={{ borderBottom: "1px solid #f0f0f0", paddingBottom: 10, marginBottom: 10 }}>
              <a href={n.url} target="_blank" rel="noreferrer" style={{ color: "#3182ce", fontWeight: 600, fontSize: 13 }}>
                {n.title}
              </a>
              {n.abstract && (
                <p style={{ fontSize: 12, color: "#4a5568", marginTop: 4 }}>
                  {n.abstract}…
                </p>
              )}
              <span style={{ fontSize: 11, color: "#a0aec0" }}>{n.date}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
import { useEffect, useRef, useState } from "react";

const SYSTEM_PROMPT = "You are MedAssist, a knowledgeable and empathetic general medical AI assistant designed for doctors and medical professionals. You assist with: general medical questions, drug interactions and dosages, differential diagnosis support, treatment protocols, medical terminology, and recent clinical research across ALL medical specialties (not limited to cardiology). Answer in the same language as the user. Be concise, evidence-based, professional. Always note that clinical judgment is paramount and AI responses are decision support only.";

const INITIAL_MESSAGE = {
  role: "assistant",
  content: "Bonjour ! Je suis MedAssist, votre assistant médical général. Comment puis-je vous aider ?",
};

export default function MedAssistChat({ apiFetch }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([INITIAL_MESSAGE]);
  const [isTyping, setIsTyping] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      inputRef.current?.focus();
    }
  }, [isOpen, messages, isTyping]);

  const sendMessage = async () => {
    const text = inputValue.trim();
    if (!text || isTyping) {
      return;
    }

    const nextMessages = [...messages, { role: "user", content: text }];
    setMessages(nextMessages);
    setInputValue("");
    setIsTyping(true);

    try {
      const data = await apiFetch("/api/ai", {
        method: "POST",
        body: JSON.stringify({
          message: text,
          messages: nextMessages.filter((m) => m.role !== "system").map((m) => ({
            role: m.role,
            content: m.content,
          })),
          system_prompt: SYSTEM_PROMPT,
          pathology: null,
          patient_context: null,
        }),
      });

      const generatedText = Array.isArray(data) ? data[0]?.generated_text : data?.generated_text;
      setMessages((prev) => [...prev, { role: "assistant", content: generatedText || "❌ Service temporairement indisponible." }]);
    } catch {
      setMessages((prev) => [...prev, { role: "assistant", content: "❌ Service temporairement indisponible." }]);
    } finally {
      setIsTyping(false);
    }
  };

  const onKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      <style>{`
        @keyframes medassist-bounce {
          0%, 80%, 100% { transform: scale(0.7); opacity: 0.45; }
          40% { transform: scale(1); opacity: 1; }
        }
      `}</style>

      <button
        type="button"
        onClick={() => setIsOpen((open) => !open)}
        aria-label="Ouvrir MedAssist"
        style={{
          position: "fixed",
          bottom: 100,
          right: 24,
          width: 56,
          height: 56,
          borderRadius: "50%",
          border: "none",
          cursor: "pointer",
          zIndex: 9999,
          color: "#ffffff",
          fontSize: 24,
          boxShadow: "0 12px 28px rgba(2, 132, 199, 0.45)",
          background: "linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)",
        }}
      >
        🩺
      </button>

      {isOpen && (
        <div
          style={{
            position: "fixed",
            bottom: 170,
            right: 24,
            width: 380,
            height: 520,
            zIndex: 9999,
            overflow: "hidden",
            borderRadius: 16,
            boxShadow: "0 20px 60px rgba(0,0,0,0.2)",
            background: "#ffffff",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              height: 60,
              padding: "12px 16px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              color: "#ffffff",
              background: "linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)",
            }}
          >
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 18, fontWeight: 800, lineHeight: 1.2 }}>🩺 MedAssist</div>
              <div style={{ fontSize: 14, opacity: 0.92, marginTop: 2 }}>Assistant Médical Général</div>
            </div>
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              aria-label="Fermer MedAssist"
              style={{
                width: 30,
                height: 30,
                border: "none",
                borderRadius: "50%",
                cursor: "pointer",
                background: "rgba(255,255,255,0.18)",
                color: "#ffffff",
                fontSize: 18,
                lineHeight: 1,
              }}
            >
              ✕
            </button>
          </div>

          <div
            style={{
              flex: 1,
              overflowY: "auto",
              padding: 14,
              background: "#ffffff",
            }}
          >
            {messages.map((message, index) => {
              const isUser = message.role === "user";
              return (
                <div
                  key={`${message.role}-${index}-${message.content.slice(0, 12)}`}
                  style={{
                    display: "flex",
                    justifyContent: isUser ? "flex-end" : "flex-start",
                    marginBottom: 10,
                  }}
                >
                  <div
                    style={{
                      maxWidth: "82%",
                      padding: "10px 12px",
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                      fontSize: 15,
                      lineHeight: 1.5,
                      color: isUser ? "#ffffff" : "#1e293b",
                      background: isUser ? "#0ea5e9" : "#f1f5f9",
                      borderRadius: isUser ? "12px 12px 2px 12px" : "12px 12px 12px 2px",
                    }}
                  >
                    {message.content}
                  </div>
                </div>
              );
            })}

            {isTyping && (
              <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 10 }}>
                <div
                  style={{
                    padding: "12px 14px",
                    background: "#f1f5f9",
                    borderRadius: "12px 12px 12px 2px",
                    display: "flex",
                    gap: 6,
                    alignItems: "center",
                  }}
                >
                  <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#0ea5e9", animation: "medassist-bounce 1s infinite" }} />
                  <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#0ea5e9", animation: "medassist-bounce 1s infinite 0.15s" }} />
                  <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#0ea5e9", animation: "medassist-bounce 1s infinite 0.3s" }} />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div
            style={{
              height: 60,
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "10px 12px",
              borderTop: "1px solid #e2e8f0",
              background: "#ffffff",
            }}
          >
            <input
              ref={inputRef}
              value={inputValue}
              onChange={(event) => setInputValue(event.target.value)}
              onKeyDown={onKeyDown}
              disabled={isTyping}
              placeholder="Posez votre question médicale..."
              style={{
                flex: 1,
                height: 40,
                borderRadius: 10,
                border: "1px solid #cbd5e1",
                padding: "0 12px",
                fontSize: 15,
                outline: "none",
                color: "#0f172a",
                background: isTyping ? "#f8fafc" : "#ffffff",
              }}
            />
            <button
              type="button"
              onClick={sendMessage}
              disabled={isTyping || !inputValue.trim()}
              style={{
                height: 40,
                padding: "0 14px",
                border: "none",
                borderRadius: 8,
                cursor: isTyping || !inputValue.trim() ? "not-allowed" : "pointer",
                background: "#0ea5e9",
                color: "#ffffff",
                fontWeight: 700,
                fontSize: 13,
                opacity: isTyping || !inputValue.trim() ? 0.65 : 1,
              }}
            >
              Envoyer
            </button>
          </div>
        </div>
      )}
    </>
  );
}
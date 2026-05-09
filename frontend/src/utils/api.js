/**
 * API utilities for RhythmAI frontend
 * Handles HTTP requests, authentication, and AI chat interactions
 */

export const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8001";

// ── Token management ──────────────────────────────────────────────────────────
export const token = () => localStorage.getItem("ra_token");

// ── API Fetch wrapper with Authorization ──────────────────────────────────────
export async function apiFetch(path, opts = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...opts,
    headers: {
      ...(opts.headers || {}),
      ...(token() ? { Authorization: `Bearer ${token()}` } : {}),
      ...(opts.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.status === 204 ? null : res.json();
}

// ── AI Chat function ──────────────────────────────────────────────────────────
export async function askAI(messages, systemPrompt, extras = {}) {
  const normalizedMessages = messages
    .filter((m) => m && m.role && m.content)
    .map((m) => ({ role: m.role, content: m.content }));
  const lastUserMessage = [...normalizedMessages].reverse().find((m) => m.role === "user")?.content || "";

  try {
    const data = await apiFetch("/api/ai", {
      method: "POST",
      body: JSON.stringify({
        message: lastUserMessage,
        messages: normalizedMessages,
        system_prompt: systemPrompt,
        pathology: extras.pathology,
        patient_context: extras.patientContext,
      }),
    });

    const generatedText = Array.isArray(data)
      ? data?.[0]?.generated_text
      : data?.generated_text;
    return generatedText || "No response";
  } catch (e) {
    console.error(e);
    throw e;
  }
}

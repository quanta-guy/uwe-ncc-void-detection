/**
 * Report assistant - a local Ollama model answering questions about one report.
 *
 * The model is a reader, not a measurer. Every figure it can cite is injected as JSON
 * built from the same evidence the report displays and exports, and the system prompt
 * forbids computing, estimating or inferring anything not present. That boundary is
 * the same one the rest of the application keeps: evaluation.py produces measurements,
 * everything else renders them.
 *
 * Why it matters here: a report answers "what" precisely and "why" not at all. A
 * reviewer asking "why did this sample fail and which field drove it" currently has to
 * read four panels and know the severity rule. That question is answerable directly
 * from data already on the page.
 *
 * Runs entirely locally through the Vite proxy to 127.0.0.1:11434. No inspection data
 * leaves the machine, which is what makes it usable on real material.
 */

import { useEffect, useRef, useState } from "react";
import { Bot, Loader2, RefreshCw, SendHorizontal } from "lucide-react";
import type { Fixtures, Inspection, ReviewEvent } from "../types";
import { fieldStatus } from "../api";

interface Msg { role: "user" | "assistant"; content: string }

const SUGGESTED = [
  "Why did this sample fail, and which field drove it?",
  "Which fields need review first, and why each one?",
  "What is the largest void, and how close is the worst field to the limit?",
  "Summarise this report for a production engineer in four sentences.",
];

/**
 * The grounding context. Deliberately the numbers, not prose: a compact JSON the model
 * can quote from. Field entries are trimmed to what a question could reasonably need,
 * so a 16-field sample stays well inside a small model's context.
 */
function buildContext(
  insp: Inspection, fx: Fixtures, status: Record<string, ReviewEvent>,
): string {
  return JSON.stringify({
    sample: {
      id: insp.sampleId, inspection: insp.inspectionId, material: insp.material,
      materialSource: "operator-entered, never inferred from the image",
      calibrationUmPerPixel: insp.umPerPixel, fields: insp.fieldCount,
    },
    disposition: {
      result: insp.preliminary.disposition, reason: insp.preliminary.reason,
      status: "preliminary - model-derived, human review pending",
    },
    profile: {
      name: fx.profile.name,
      clusterSeverityLimitUm: fx.profile.clusterSeverityLimitUm,
      mergeDistanceUm: fx.profile.mergeDistanceUm,
      approvedForProductionUse: fx.profile.approvedForProductionUse,
      severityRule:
        "cluster severity = sum of void Feret diameters + 0.5 * sqrt(total void area). "
        + "Voids closer than the merge distance count as one defect.",
    },
    model: { id: fx.model.id, threshold: fx.model.threshold, minSize: fx.model.minSize },
    sampleKpis: insp.kpis,
    controllingFieldId: insp.controllingFieldId,
    fields: insp.fields.map((f) => ({
      id: f.id,
      clusterSeverityUm: f.clusterSeverityUm,
      distanceToLimitUm: f.distanceToLimitUm,
      voidCount: f.voidCount,
      largestFeretUm: f.largestFeretUm,
      voidArealFractionPct: f.voidArealFractionPct,
      verdict: f.verdict,
      triage: f.triageAction,
      reason: f.triageReason,
      modelDisagreement: f.modelDisagreement,
      reviewStatus: fieldStatus(status, insp.inspectionId, f.id),
      controllingDefect: f.severityEvidence.voids.length
        ? {
            voids: f.severityEvidence.voids.map((v) => ({
              label: v.label, lengthUm: v.lengthUm, areaUm2: v.areaUm2,
            })),
            mergeGapsUm: f.severityEvidence.gaps.map((g) => g.gapUm),
            lengthTermUm: f.severityEvidence.length_term_um,
            areaTermUm: f.severityEvidence.area_term_um,
          }
        : null,
    })),
  });
}

const SYSTEM = `You are a quality-inspection assistant reading ONE cross-section report.

RULES, in order of importance:
1. Answer only from the REPORT DATA below. If something is not in it, say "that is not in this report" and stop.
2. Never calculate, estimate, average or infer a measurement. Quote figures exactly as given, with their units.
3. Every number you state must appear verbatim in the data. If you cannot find it, say so.
4. Field numbers are two digits, e.g. "field 12".
5. All results are preliminary and model-derived until a human reviews them. Say so when giving a disposition.
6. Be brief: 2-5 sentences unless asked for more. No preamble, no markdown headings.
7. You are not authorised to approve, reject or certify anything. You describe what the report says.`;

/**
 * Reasoning models think out loud before answering, and an inspection tool must not
 * show that. Two shapes occur in practice: a properly paired <think>...</think> block,
 * and - with Qwen3 through Ollama, even when `think: false` is requested - a closing
 * </think> with no opening tag, because the daemon strips the opener.
 *
 * Anything before a closing tag is reasoning and is dropped. While reasoning is still
 * streaming there is no answer yet, so the caller shows a working indicator rather than
 * a wall of deliberation a reviewer would have to read past.
 */
const REASONING_MODEL = /qwen3|deepseek-r1|qwq|magistral|\br1\b/i;

export function visible(raw: string, reasoning = false): string {
  const close = raw.lastIndexOf("</think>");
  if (close !== -1) return raw.slice(close + "</think>".length).trimStart();
  if (raw.includes("<think>")) return "";
  // A reasoning model that has not closed its block yet is still deliberating, and
  // there is no way to tell its thoughts from its answer until the tag arrives.
  if (reasoning) return "";
  return raw.trimStart();
}

interface Props { inspection: Inspection; fx: Fixtures; status: Record<string, ReviewEvent> }

export function ReportChat({ inspection, fx, status }: Props) {
  const [models, setModels] = useState<string[] | null>(null);
  const [model, setModel] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    fetch("/ollama/api/tags")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d: { models?: Array<{ name: string }> }) => {
        const names = (d.models ?? []).map((m) => m.name);
        setModels(names);
        // Prefer a small general instruct model; coder models answer this badly.
        setModel(names.find((n) => /qwen3:|llama3|mistral|gemma/i.test(n)) ?? names[0] ?? "");
      })
      .catch(() => setModels([]));
  }, []);

  useEffect(() => { setMsgs([]); }, [inspection.inspectionId]);
  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [msgs, busy]);

  const ask = async (question: string) => {
    if (!question.trim() || busy || !model) return;
    const history = [...msgs, { role: "user" as const, content: question }];
    setMsgs([...history, { role: "assistant", content: "" }]);
    setInput("");
    setBusy(true);
    setErr(null);
    const thinks = REASONING_MODEL.test(model);
    abort.current = new AbortController();

    try {
      const res = await fetch("/ollama/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abort.current.signal,
        body: JSON.stringify({
          model,
          // Qwen3 emits reasoning blocks unless asked not to; a report answer does not
          // need them and they read as noise in an inspection tool.
          think: false,
          stream: true,
          options: { temperature: 0.1, num_ctx: 8192 },
          messages: [
            { role: "system", content: `${SYSTEM}\n\nREPORT DATA:\n${buildContext(inspection, fx, status)}` },
            ...history,
          ],
        }),
      });
      if (!res.ok || !res.body) throw new Error(`Ollama returned ${res.status}`);

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let acc = "";
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const j = JSON.parse(line) as { message?: { content?: string } };
            acc += j.message?.content ?? "";
          } catch { /* partial line, picked up next chunk */ }
        }
        setMsgs([...history, { role: "assistant", content: visible(acc, thinks) }]);
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setErr((e as Error).message);
        setMsgs(history);
      }
    } finally {
      setBusy(false);
    }
  };

  /** Header is shared by every state, so the panel looks the same before a model loads. */
  const Head = ({ subtitle }: { subtitle?: React.ReactNode }) => (
    <div className="chat-head">
      <div className="row">
        <span className="chat-avatar"><Bot size={21} aria-hidden /></span>
        <div>
          <div className="chat-title">Report assistant</div>
          {subtitle && <div className="chat-sub">{subtitle}</div>}
        </div>
      </div>
      <svg className="chat-wave" viewBox="0 0 1200 40" preserveAspectRatio="none" aria-hidden>
        <path d="M0,40 C300,0 900,0 1200,40 L1200,40 L0,40 Z" fill="#fff" />
      </svg>
    </div>
  );

  if (models === null) {
    return (
      <div className="chat">
        <Head subtitle="Looking for a local model…" />
        <div className="chat-body">
          <div className="bubble bot">Checking for a model on this machine…</div>
        </div>
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className="chat">
        <Head subtitle="No local model reachable" />
        <div className="chat-body">
          <div className="note warn">
            No local model is reachable. Start Ollama and pull a small instruct model:
            <div className="mono" style={{ marginTop: 8 }}>ollama serve</div>
            <div className="mono">ollama pull qwen3:4b</div>
            <div style={{ marginTop: 8 }}>
              The assistant runs entirely on this machine — no inspection data leaves it.
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="chat">
      <div className="chat-head">
        <div className="spread">
          <div className="row">
            <span className="chat-avatar"><Bot size={21} aria-hidden /></span>
            <div>
              <div className="chat-title">Report assistant</div>
              <div className="chat-sub">
                <span className="chat-online" aria-hidden /> Running locally · {model || "no model"}
              </div>
            </div>
          </div>
          <div className="row" style={{ gap: 8 }}>
            <select value={model} onChange={(e) => setModel(e.target.value)}
                    aria-label="Model" style={{ width: "auto", minHeight: 34, fontSize: 13 }}>
              {models.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
            {msgs.length > 0 && (
              <button className="btn sm" onClick={() => { abort.current?.abort(); setMsgs([]); }}>
                <RefreshCw size={14} aria-hidden /> Clear
              </button>
            )}
          </div>
        </div>
        {/* Curve into the message area, so the header reads as a distinct surface. */}
        <svg className="chat-wave" viewBox="0 0 1200 40" preserveAspectRatio="none" aria-hidden>
          <path d="M0,40 C300,0 900,0 1200,40 L1200,40 L0,40 Z" fill="#fff" />
        </svg>
      </div>

      <div className="chat-body" ref={scroller}>
        {msgs.length === 0 && (
          <>
            <div className="bubble bot" style={{ maxWidth: "100%" }}>
              Ask me about <strong>{inspection.sampleId}</strong> — {inspection.fieldCount} fields,
              {" "}{inspection.material}. I quote the measurements on this page and will tell you
              when something is not in the report. I cannot compute a new measurement, and I
              cannot approve or reject a sample.
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, margin: "14px 0 4px" }}>
              {SUGGESTED.map((q) => (
                <button key={q} className="chat-q" onClick={() => ask(q)}>{q}</button>
              ))}
            </div>
          </>
        )}

        {msgs.map((m, i) => (
          <div key={i} className={`bubble ${m.role === "user" ? "me" : "bot"}`}>
            {m.content || (busy && i === msgs.length - 1
              ? <span className="row" style={{ gap: 8, color: "var(--muted)" }}>
                  <Loader2 size={15} className="spin" aria-hidden />
                  {REASONING_MODEL.test(model) ? "reasoning over the report…" : "reading the report…"}
                </span>
              : null)}
          </div>
        ))}

        {err && <div className="note bad" style={{ marginTop: 10 }}>Model error: {err}</div>}
      </div>

      <div className="chat-foot">
        <form className="chat-input" onSubmit={(e) => { e.preventDefault(); ask(input); }}>
          <input type="text" value={input} disabled={busy}
                 placeholder={`Ask about ${inspection.sampleId}…`}
                 aria-label="Ask about this report"
                 onChange={(e) => setInput(e.target.value)} />
          <button className="send" type="submit" disabled={busy || !input.trim()}
                  aria-label="Send question">
            {busy ? <Loader2 size={18} className="spin" aria-hidden />
                  : <SendHorizontal size={18} aria-hidden />}
          </button>
        </form>
        <div className="provisional" style={{ marginTop: 10 }}>
          Generated from this report's data by a local model. Measurements come from
          evaluation.py, not from the model — check any figure against the panels above
          before acting on it.
        </div>
      </div>
    </div>
  );
}

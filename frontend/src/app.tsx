/**
 * Application shell and routes.
 *
 * Fixtures load once and are handed down by context; a query library would be more
 * state machinery than a filesystem-backed prototype needs. Review events live in
 * localStorage and are re-read on a custom event, so every screen recalculates
 * together when a decision is recorded.
 */

import { createContext, useContext, useEffect, useState } from "react";
import {
  BrowserRouter, NavLink, Navigate, Route, Routes,
} from "react-router-dom";
import {
  ClipboardList, FileText, LayoutList, Settings as SettingsIcon, Sparkles,
} from "lucide-react";
import { loadFixtures, readReviews, statusMap } from "./api";
import type { Fixtures, ReviewEvent } from "./types";
import { InspectionsPage } from "./pages/InspectionsPage";
import { SampleAnalysisPage } from "./pages/SampleAnalysisPage";
import { FieldReviewPage } from "./pages/FieldReviewPage";
import { ReportsPage } from "./pages/ReportsPage";
import { CrossSectionReportPage } from "./pages/CrossSectionReportPage";
import { ModelImprovementPage } from "./pages/ModelImprovementPage";
import { SettingsPage } from "./pages/SettingsPage";

interface Ctx {
  fx: Fixtures;
  reviews: ReviewEvent[];
  status: Record<string, ReviewEvent>;
}
const DataCtx = createContext<Ctx | null>(null);

export function useData(): Ctx {
  const c = useContext(DataCtx);
  if (!c) throw new Error("useData outside provider");
  return c;
}

const NAV = [
  { to: "/inspections", label: "Inspections", Icon: LayoutList },
  { to: "/review", label: "Review queue", Icon: ClipboardList },
  { to: "/reports", label: "Reports", Icon: FileText },
  { to: "/model-improvement", label: "Model improvement", Icon: Sparkles },
  { to: "/settings", label: "Settings", Icon: SettingsIcon },
];

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <>
      <nav className="nav" aria-label="Main">
        <span className="brand">Composite Inspection</span>
        {NAV.map(({ to, label, Icon }) => (
          <NavLink key={to} to={to} className={({ isActive }) => (isActive ? "active" : "")}>
            <Icon size={17} aria-hidden /> {label}
          </NavLink>
        ))}
      </nav>
      {children}
    </>
  );
}

export default function App() {
  const [fx, setFx] = useState<Fixtures | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [reviews, setReviews] = useState<ReviewEvent[]>(readReviews());

  useEffect(() => {
    loadFixtures().then(setFx).catch((e: Error) => setErr(e.message));
  }, []);

  useEffect(() => {
    const sync = () => setReviews(readReviews());
    window.addEventListener("cip:reviews", sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener("cip:reviews", sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  // BrowserRouter wraps everything, including the loading and error states: Shell
  // renders NavLink, and a NavLink outside a Router throws rather than degrading.
  if (err) {
    return (
      <BrowserRouter>
        <Shell>
          <div className="page">
            <div className="note bad">
              <strong>Could not load inspection data.</strong>
              <div style={{ marginTop: 8 }} className="mono">{err}</div>
              <div style={{ marginTop: 10 }}>
                Generate it with{" "}
                <span className="mono">python frontend/tools/build_fixtures.py</span>
              </div>
            </div>
          </div>
        </Shell>
      </BrowserRouter>
    );
  }
  if (!fx) {
    return (
      <BrowserRouter>
        <Shell>
          <div className="page"><div className="empty">Loading inspection data…</div></div>
        </Shell>
      </BrowserRouter>
    );
  }

  return (
    <DataCtx.Provider value={{ fx, reviews, status: statusMap(reviews) }}>
      <BrowserRouter>
        <Shell>
          <Routes>
            <Route path="/" element={<Navigate to="/inspections" replace />} />
            <Route path="/inspections" element={<InspectionsPage />} />
            <Route path="/inspections/:inspectionId" element={<SampleAnalysisPage />} />
            <Route path="/inspections/:inspectionId/fields/:fieldId" element={<FieldReviewPage />} />
            <Route path="/review" element={<FieldReviewPage />} />
            <Route path="/reports" element={<ReportsPage />} />
            <Route path="/reports/:inspectionId" element={<CrossSectionReportPage />} />
            <Route path="/model-improvement" element={<ModelImprovementPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<div className="page"><div className="empty">Not found.</div></div>} />
          </Routes>
        </Shell>
      </BrowserRouter>
    </DataCtx.Provider>
  );
}

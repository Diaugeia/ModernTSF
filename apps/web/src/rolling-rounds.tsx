"use client";

import type { RollingSummary } from "./lib/leaderboard-data";
import { RankBadge, Th } from "./ui/table-bits";

export interface RollingCopy {
  title: string;
  intro: string;
  rounds: string;
  stability: string;
  meanRank: string;
  meanMse: string;
  empty: string;
  submit: string;
}

const fmt = (v: number | null | undefined, digits = 4) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);

/** Weekly rounds: forecasts are submitted before their truth exists, then scored. */
export function RollingRounds({ data, copy }: { data: Record<string, RollingSummary>; copy: RollingCopy }) {
  const tracks = Object.values(data);
  return (
    <section className="mt-8 rounded-xl border border-border bg-surface p-5">
      <h2 className="text-base font-semibold text-ink">{copy.title}</h2>
      <p className="mt-1 max-w-3xl text-sm text-muted">
        {copy.intro}{" "}
        <a className="text-accent underline-offset-2 hover:underline"
           href="https://github.com/Diaugeia/ModernTSF/blob/main/apps/web/SUBMITTING.md#real-time-rounds">
          {copy.submit}
        </a>
      </p>
      {tracks.length === 0 && <p className="mt-4 text-sm text-faint">{copy.empty}</p>}
      {tracks.map((summary) => (
        <div key={summary.track} className="mt-5">
          <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-sm">
            <span className="font-medium text-ink">{summary.track}</span>
            <span className="text-muted">{copy.rounds}: {summary.scored_rounds.length}</span>
            <span className="text-muted">{copy.stability}: {fmt(summary.rank_stability.mean, 2)}</span>
          </div>
          {summary.methods.length === 0 ? (
            <p className="mt-2 text-sm text-faint">{copy.empty}</p>
          ) : (
            <div className="mt-2 overflow-x-auto">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr>
                    <Th>#</Th>
                    <Th>Model</Th>
                    <Th>{copy.rounds}</Th>
                    <Th>{copy.meanRank}</Th>
                    <Th>{copy.meanMse}</Th>
                  </tr>
                </thead>
                <tbody>
                  {summary.methods.map((m, i) => (
                    <tr key={m.model} className="border-b border-border/60">
                      <td className="px-5 py-2"><RankBadge rank={i + 1} /></td>
                      <td className="px-5 py-2 text-ink">{m.model}</td>
                      <td className="px-5 py-2 text-muted [font-variant-numeric:tabular-nums]">{m.rounds}</td>
                      <td className="px-5 py-2 text-muted [font-variant-numeric:tabular-nums]">{fmt(m.mean_rank, 2)}</td>
                      <td className="px-5 py-2 text-muted [font-variant-numeric:tabular-nums]">{fmt(m.mean_mse)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ))}
    </section>
  );
}

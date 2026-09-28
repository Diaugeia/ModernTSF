// Shape of data/leaderboard.json (produced by the TSEval build pipeline).
import type { ModelType } from "./model-types";

export interface LeaderRow {
  model: string;
  mse: number | null;
  mae: number | null;
  rmse?: number | null;
  wape?: number | null;
  rse?: number | null;
  corr: number | null;
  n_runs: number;
  submission_ids: string[];
  rank?: number;
  displayRank?: number;
  categoryRank?: number;
  modelType?: ModelType;
  // Quant (trading) metrics — present on stock-track quant rows.
  total_return?: number | null;
  annualized_return?: number | null;
  sharpe?: number | null;
  max_drawdown?: number | null;
  win_rate?: number | null;
  avg_turnover?: number | null;
}

export interface DatasetBlock {
  horizons: Record<string, LeaderRow[]>;
  // Stock track only: backtest_91d_{conservative|balanced|aggressive} → rows.
  quant?: Record<string, LeaderRow[]>;
}

export interface TrackBlock {
  datasets: Record<string, DatasetBlock>;
}

export interface LeaderboardData {
  schema_version: string;
  generated_at: string;
  primary_metric: string;
  n_submissions: number;
  n_rejected: number;
  tracks: Record<string, TrackBlock>;
}

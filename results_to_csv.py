"""Exporta runs do MLflow para CSV.

Uso:
  python results_to_csv.py --experiment "Nome" --out CSV/experiment.csv
  python results_to_csv.py --experiment-id 3 --out CSV/exp3.csv

Requer:
  pip install mlflow pandas
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence

import mlflow
import pandas as pd


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Achata colunas MultiIndex (vindas de metrics/params/tags) em colunas simples."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [".".join([str(x) for x in col if x != ""]) for col in df.columns.values]
    return df


def _filter_by_start_date(df: pd.DataFrame, start_date: Optional[str]) -> pd.DataFrame:
    """Filtra runs com start_time >= start_date (YYYY-MM-DD)."""
    if not start_date:
        return df
    if "start_time" not in df.columns:
        return df
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    return df[df["start_time"] >= start_ms]


def export_mlflow_to_csv(
    experiment_ids: Iterable[str],
    out_path: str,
    max_results: int = 100000,
    include_deleted: bool = False,
    start_date: Optional[str] = None,
) -> str:
    """Busca runs dos experimentos e salva em CSV.

    Args:
        experiment_ids: IDs dos experimentos MLflow.
        out_path: caminho do arquivo CSV de saída.
        max_results: limite de runs a buscar por experimento.
        include_deleted: inclui runs deletadas se True.

    Returns:
        Caminho do CSV gerado.
    """
    runs: List[pd.DataFrame] = []

    for exp_id in experiment_ids:
        df = mlflow.search_runs(
            experiment_ids=[exp_id],
            max_results=max_results,
            run_view_type=mlflow.entities.ViewType.ALL
            if include_deleted
            else mlflow.entities.ViewType.ACTIVE_ONLY,
        )
        df = _flatten_columns(df)
        df = _filter_by_start_date(df, start_date)
        runs.append(df)

    if not runs:
        raise ValueError("Nenhum experimento encontrado ou sem runs.")

    all_runs = pd.concat(runs, ignore_index=True) if len(runs) > 1 else runs[0]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    all_runs.to_csv(out_path, index=False)
    return out_path


def _resolve_experiment_ids(
    name: Optional[str],
    exp_id: Optional[str],
    name_contains: Sequence[str],
) -> List[str]:
    if exp_id:
        return [exp_id]
    if name:
        exp = mlflow.get_experiment_by_name(name)
        if not exp:
            raise ValueError(f"Experimento '{name}' nao encontrado.")
        return [exp.experiment_id]
    # Se nenhum foi passado, exporta apenas os experimentos com o padrao no nome
    matches: List[str] = []
    for e in mlflow.search_experiments():
        exp_name = (e.name or "").upper()
        if any(token.upper() in exp_name for token in name_contains):
            matches.append(e.experiment_id)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta runs do MLflow para CSV")
    parser.add_argument("--experiment", type=str, help="Nome do experimento")
    parser.add_argument("--experiment-id", type=str, help="ID do experimento")
    parser.add_argument("--out", type=str, default="mlflow_runs.csv", help="Caminho do CSV")
    parser.add_argument(
        "--max-results",
        type=int,
        default=100000,
        help="Numero maximo de runs por experimento",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Inclui runs deletadas",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="Opcional: MLflow tracking URI (ex: sqlite:///mlflow.db ou http://...) ",
    )
    parser.add_argument(
        "--name-contains",
        type=str,
        default="AE,SKIP",
        help="Filtro por nome do experimento (tokens separados por virgula).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Apenas runs com start_time >= data (YYYY-MM-DD).",
    )
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    name_contains = [s.strip() for s in args.name_contains.split(",") if s.strip()]
    exp_ids = _resolve_experiment_ids(args.experiment, args.experiment_id, name_contains)
    out = export_mlflow_to_csv(
        exp_ids,
        args.out,
        max_results=args.max_results,
        include_deleted=args.include_deleted,
        start_date=args.start_date,
    )
    print(f"CSV gerado em: {out}")


if __name__ == "__main__":
    main()

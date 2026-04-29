"""Common base classes and mixins to reduce repetition across analyzers/plotters.

Provides AnalysisBase with:
- standard configuration (show_plot, save_data, output_dir, save_payload_dir)
- persist_plot(name, payload) helper to save plot data via save_plot_data
- resolve_output_dir() utility

Intended to minimize repeated if/else and unify behavior.
"""
from __future__ import annotations

import os
from typing import Optional, Dict, Any
from .plot_io import save_plot_data


class AnalysisBase:
    def __init__(
        self,
        *,
        show_plot: bool = True,
        save_data: bool = False,
        output_dir: Optional[str] = None,
        save_payload_dir: Optional[str] = None,
    ) -> None:
        self.show_plot = show_plot
        self.save_data = save_data
        self.output_dir = output_dir
        self.save_payload_dir = save_payload_dir

    def resolve_output_dir(self, default: str = "plot_data") -> str:
        base = self.output_dir or default
        os.makedirs(base, exist_ok=True)
        return base

    def persist_plot(self, name: str, payload: Dict[str, Any]) -> Optional[str]:
        try:
            base_dir = self.save_payload_dir or self.output_dir or "plot_data"
            return save_plot_data(name=name, payload=payload, base_dir=base_dir)
        except Exception as e:
            print(f"WARNING: persist_plot failed for {name}: {e}")
            return None

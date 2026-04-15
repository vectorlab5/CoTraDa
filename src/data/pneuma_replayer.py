from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class PneumaReplayer:
    """
    Loads and replays real-world vehicle trajectories from drone-captured datasets (like pNEUMA).
    This is used to test against realistic urban traffic patterns rather than random Brownian movement.
    """
    def __init__(self, data_path: str, area_m: tuple[float, float]):
        self.data_path = Path(data_path)
        self.area_m = area_m
        self.df: Optional[pd.DataFrame] = None
        self.time_steps: List[float] = []
        self.current_step_idx = 0
        
        if self.data_path.exists():
            self._load_data()
        else:
             logger.warning(f"pNEUMA data not found at {data_path}. Defaulting to synthetic pNEUMA-like swarm.")
             self._generate_synthetic_pneuma()

    def _load_data(self) -> None:
        """
        Reads the CSV and extracts the time step indices.
        """
        try:
            self.df = pd.read_csv(self.data_path)
            self.time_steps = sorted(self.df["time"].unique().tolist())
            logger.info(f"Loaded {len(self.df)} vehicle points from {self.data_path}")
        except Exception as e:
            logger.error(f"Failed to load pNEUMA CSV: {e}")
            self._generate_synthetic_pneuma()

    def _generate_synthetic_pneuma(self) -> None:
        """
        Fallback for when the dataset is missing. Generates a large group of 
        points moving in pseudo-urban patterns to keep the simulation running.
        """
        num_points = 5000
        ids = np.repeat(np.arange(100), 50)
        times = np.tile(np.arange(0, 50, 1.0), 100)
        xs = np.random.uniform(0, self.area_m[0], size=num_points)
        ys = np.random.uniform(0, self.area_m[1], size=num_points)
        self.df = pd.DataFrame({
            "vehicle_id": ids,
            "time": times,
            "x": xs,
            "y": ys,
            "speed": np.random.uniform(10, 50, size=num_points)
        })
        self.time_steps = sorted(self.df["time"].unique().tolist())

    def get_step(self, step_idx: int, num_agents: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the vehicle state (position, speed) for a given time step.
        If the dataset has fewer vehicles than requested, it pads the data.
        """
        t = self.time_steps[step_idx % len(self.time_steps)]
        step_df = self.df[self.df["time"] == t]
        
        # Ensure we have enough data for all agents by wrapping or padding
        if len(step_df) < num_agents:
            step_df = pd.concat([step_df] * (num_agents // len(step_df) + 1)).iloc[:num_agents]
        else:
            step_df = step_df.iloc[:num_agents]
            
        coords = step_df[["x", "y"]].values.astype(np.float32)
        speeds = step_df["speed"].values.astype(np.float32)
        return coords, speeds

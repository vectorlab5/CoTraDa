from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    num_agents: int
    num_relays: int
    area_m: tuple[int, int]
    speed_kmh: tuple[float, float]
    max_hops: int


SCENARIOS = {
    "urban_small": ScenarioSpec("urban_small", num_agents=20, num_relays=4, area_m=(500, 500), speed_kmh=(20.0, 40.0), max_hops=3),
    "urban_large": ScenarioSpec("urban_large", num_agents=50, num_relays=10, area_m=(2000, 2000), speed_kmh=(20.0, 50.0), max_hops=5),
    "highway": ScenarioSpec("highway", num_agents=30, num_relays=6, area_m=(5000, 400), speed_kmh=(80.0, 120.0), max_hops=4),
    "pneuma_athens": ScenarioSpec("pneuma_athens", num_agents=30, num_relays=4, area_m=(490, 358), speed_kmh=(10.0, 60.0), max_hops=4),
}


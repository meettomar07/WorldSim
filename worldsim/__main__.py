"""Run WorldSim's Phase 1 viewer."""

from .config import WorldConfig
from .engine import Simulation
from .viewer import WorldViewer


def main() -> None:
    WorldViewer(Simulation(WorldConfig())).run()


if __name__ == "__main__":
    main()

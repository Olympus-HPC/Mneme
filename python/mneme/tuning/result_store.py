import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from mneme.utils import MnemeEncoder


class ResultStore:
    """ Utility for storing tunin results so that we can easily reload them and restart
        experiments from a previous state.
    """

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.logs_dir = self.results_dir / "logs"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "tune.log").touch(exist_ok=True)
        self.trials_file = self.results_dir / "trials.jsonl"

    def path(self, name: str) -> Path:
        return self.results_dir / name

    def write_json(self, name: str, data: Any) -> None:
        with open(self.path(name), "w") as fd:
            json.dump(data, fd, cls=MnemeEncoder, indent=2)
            fd.write("\n")

    def write_config(self, config: Dict[str, Any]) -> None:
        self.write_json("config.json", config)

    def write_search_space(self, search_space: Dict[str, Any]) -> None:
        self.write_json("search_space.json", search_space)

    def write_baseline(self, baseline: Dict[str, Any]) -> None:
        self.write_json("baseline.json", baseline)

    def append_trial(self, trial: Dict[str, Any]) -> None:
        with open(self.trials_file, "a") as fd:
            json.dump(trial, fd, cls=MnemeEncoder, sort_keys=True)
            fd.write("\n")

    def write_best(self, best: Dict[str, Any]) -> None:
        self.write_json("best.json", best)

    def write_summary(self, summary: Dict[str, Any]) -> None:
        self.write_json("summary.json", summary)

    def load_config(self) -> Optional[Dict[str, Any]]:
        path = self.path("config.json")
        if not path.exists():
            return None
        
        with open(path, "r") as fd:
            return json.load(fd)

    def load_baseline(self) -> Optional[Dict[str, Any]]:
        path = self.path("baseline.json")
        if not path.exists():
            return None
        
        with open(path, "r") as fd:
            return json.load(fd)

    def load_trials(self) -> Iterable[Dict[str, Any]]:
        if not self.trials_file.exists():
            return []
        
        trials = []
        with open(self.trials_file, "r") as fd:
            for line in fd:
                line = line.strip()
                if not line:
                    continue
                trials.append(json.loads(line))
        
        return trials

    def completed_hashes(self) -> Set[str]:
        hashes = set()
        for trial in self.load_trials():
            config_hash = trial.get("config_hash")
            if config_hash:
                hashes.add(config_hash)
        
        return hashes

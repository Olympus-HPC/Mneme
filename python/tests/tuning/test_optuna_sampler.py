import pytest
import random
import optuna


from mneme.tuning.parameters import *
from mneme.tuning.search_space import SearchSpace


class OptunaSpace(SearchSpace):

    def dimensions(self):
        return {
            "x": IntRangeParam("x", 0, 10, step=2),
            "flag": BoolParam("flag"),
            "choice": CategoricalParam("choice", ["A", "B"]),
            "fix": FixedParam("fix", 7),
        }

    def derived(self, p):
        return {"double": p["x"] * 2}

    def constraints(self, p):
        # forbid x = 4 for testing constraint logic
        return p["x"] != 4


def test_optuna_sampling_basic():
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    space = OptunaSpace()

    cfg = space.sample_optuna(trial)

    assert cfg["x"] in {0, 2, 6, 8, 10}  # step=2 and x!=4
    assert cfg["flag"] in [True, False]
    assert cfg["choice"] in ["A", "B"]
    assert cfg["fix"] == 7
    assert cfg["double"] == cfg["x"] * 2


def test_optuna_pipeline_param_generator():

    def gen():
        return "PIPELINE_" + str(random.randint(0, 10))

    class PipelineSpace(SearchSpace):
        def dimensions(self):
            return {"pipeline": PipelineParam("pipeline", generator=gen)}

    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    space = PipelineSpace()
    cfg = space.sample_optuna(trial)

    assert cfg["pipeline"].startswith("PIPELINE_")

    # Ensure stored in user attrs
    assert trial.user_attrs["pipeline"] == cfg["pipeline"]

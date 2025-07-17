import json
import math
import random

from mneme.experiment import Experiment
from optuna.samplers import (
    GPSampler,
    NSGAIISampler,
    QMCSampler,
    RandomSampler,
    TPESampler,
)


class BuildConfigWrapper:
    def __init__(self, LLVMPipelineManager, KernelDescr):
        self._manager = LLVMPipelineManager
        max_threads = int(
            KernelDescr.block_dim.x * KernelDescr.block_dim.y * KernelDescr.block_dim.z
        )
        self.max_threads = max_threads

        self.min_blocks_per_sm_max = int(math.ceil(1024 / (max_threads * 2)))

    def __call__(self, executor, trial):
        indexes = []
        for v in range(0, 160):
            indexes.append(trial.suggest_int(f"pass_id_{v}", -40, 116))
        pipeline = []
        for v in indexes:
            # Everything below 0, we ignore
            if v >= 0:
                pipeline.append(self._manager._passes[v].optuna_pass(trial, f"pos:{v}"))

        min_blocks_per_sm = trial.suggest_int(
            "min_blocks_per_sm", 0, self.min_blocks_per_sm_max
        )
        max_threads = self.max_threads
        if min_blocks_per_sm == 0:
            max_threads = 0

        return Experiment(
            specialize=executor.specialize,
            max_threads=max_threads,
            min_blocks_per_sm=min_blocks_per_sm,
            specialize_dims=executor.specialize,
            passes=self._manager.to_string(pipeline),
            prune=executor.prune,
            internalize=executor.internalize,
            codegen_opt=executor.codegen_opt,
            rtc=executor.rtc,
            device_arch=executor.device_arch,
        )


def schedule_job(
    db,
    study,
    executor,
    ConfigWrapper,
    performed_experiments,
    num_trials,
    exp_id,
    worker,
):
    while True:
        if performed_experiments < num_trials:
            trial = study.ask()
            e = ConfigWrapper(executor, trial)
            # TODO: We should implement skipping, but we need to register (tell optuna) about the result.
            if not db.should_execute(e):
                print(f"Skipping experiment {e.hash()}")
                continue
            worker.assign(
                {
                    "payload": "process",
                    "data": e.to_dict(),
                    "exp_id": exp_id + 1,
                }
            )
            return (e, trial, exp_id + 1)
        else:
            return None


def get_sampler(sampler_name, seed):
    if sampler_name == "RandomSampler":
        return RandomSampler(seed=seed)
    elif sampler_name == "TPESampler":
        return TPESampler(seed=seed)
    elif sampler_name == "GPSampler":
        return GPSampler(seed=seed)
    elif sampler_name == "NSGAIISampler":
        return NSGAIISampler(seed=seed)
    elif sampler_name == "QMCSampler":
        return QMCSampler(seed=seed)
    else:
        raise RuntimeError("Unsupported sampling technique")


def run_optuna_tune(
    custom_db,
    executor,
    workers,
    completed_jobs_q,
    num_trials,
    sampler_name,
    seed,
):
    import optuna
    from optuna.trial import TrialState

    _filename = f"{custom_db.db_dir}/mneme.{executor.kernel_descr.static_hash}.{executor.kernel_descr.dynamic_hash}"
    if executor.suffix is not None:
        _filename += f".{executor.suffix}"

    sampler = get_sampler(sampler_name, seed)

    db_path = f"sqlite:////{_filename}.{sampler_name}.{seed}.sql"
    print(db_path)
    study = optuna.create_study(
        sampler=sampler,
        storage=db_path,
        study_name=f"{executor.kernel_descr.static_hash}.{executor.kernel_descr.dynamic_hash}",
        direction="minimize",
        load_if_exists=True,
    )
    print(f"Existing trial count: {len(study.trials)}")
    performed_experiments = len(study.trials)
    num_in_process = 0
    exp_id = performed_experiments

    if performed_experiments >= num_trials:
        return
    Configure = BuildConfigWrapper(executor.LLVMPassManager, executor.kernel_descr)

    root_ir = executor.link_ir()
    orig = custom_db.save_ir(str(root_ir), "orig")

    in_flight = {}

    for w in workers:
        vals = schedule_job(
            custom_db,
            study,
            executor,
            Configure,
            performed_experiments + num_in_process,
            num_trials,
            exp_id,
            w,
        )
        if vals is None:
            continue
        exp, trial, exp_id = vals[0], vals[1], vals[2]
        print("Experiment id is", exp_id)
        in_flight[exp_id] = (exp, w, trial)
        num_in_process += 1

    while len(in_flight) != 0:
        res = completed_jobs_q.get()
        if res["payload"] == "result":
            done_exp_id = res["exp_id"]
            exp1, worker, trial = in_flight.pop(done_exp_id)
            exp2 = Experiment.from_dict(**res["data"])
            num_in_process -= 1
            performed_experiments += 1
            if exp1.hash() != exp2.hash():
                exp1.dump()
                exp2.dump()
                raise RuntimeError(
                    f"Received experiment should have same hash with workers experiment {exp1.hash()} {exp2.hash()}"
                )
            print(
                f"Worker {worker.idx} Done with {done_exp_id} had {exp2.failed} and took {exp2.exec_time}"
            )
            if not exp2.failed:
                if "llvm_ir" not in res:
                    raise RuntimeError(
                        "Expected llvm ir to exist on non failed experiment"
                    )
                final = custom_db.save_ir(res["llvm_ir"], exp2.hash())
                custom_db.add(orig, final, exp2)
                for k, v in res["data"].items():
                    trial.set_user_attr(k, v)
                study.tell(trial, values=exp2.exec_time)
            else:
                custom_db.add(orig, "Error", exp2)
                study.tell(trial, state=TrialState.FAIL)

            vals = schedule_job(
                custom_db,
                study,
                executor,
                Configure,
                performed_experiments + num_in_process,
                num_trials,
                exp_id,
                worker,
            )
            if vals is None:
                continue
            exp, trial, exp_id = vals[0], vals[1], vals[2]
            in_flight[exp_id] = (exp, worker, trial)
            num_in_process += 1
        else:
            print(json.dumps(res, indent=6))
            raise RuntimeError("Unknown paylod")

    print("Going back")

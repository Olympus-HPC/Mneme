import hashlib
import json


class Experiment:
    def __init__(self, **kwargs):
        self._specialize = kwargs.pop("specialize")
        self._max_threads = kwargs.pop("max_threads")
        self._min_blocks_per_sm = kwargs.pop("min_blocks_per_sm")
        self._specialize_dims = kwargs.pop("specialize_dims")
        self._passes = kwargs.pop("passes")
        self._prune = kwargs.pop("prune")
        self._internalize = kwargs.pop("internalize")
        self._codegen_opt = kwargs.pop("codegen_opt")
        self._rtc = kwargs.pop("rtc")
        self._device_arch = kwargs.pop("device_arch")

        self._opt_time = kwargs.pop("opt_time", None)
        self._codegen_time = kwargs.pop("codegen_time", None)
        self._verified = kwargs.pop("verified", None)
        self._obj_size = kwargs.pop("obj_size", None)
        self._exec_time = kwargs.pop("exec_time", None)
        self._executed = kwargs.pop("executed", False)
        self._failed = kwargs.pop("failed", False)

    @property
    def exec_time(self):
        return self._opt_time

    @exec_time.setter
    def exec_time(self, value):
        if not isinstance(value, list):
            raise TypeError(
                f"Optimization time expects a list of values {value}, {type(value)}"
            )

        self._exec_time = sum(value) / len(value)

    @property
    def obj_size(self):
        return self._obj_size

    @obj_size.setter
    def obj_size(self, value):
        self._obj_size = value

    @property
    def verified(self):
        return self._verified

    @verified.setter
    def verified(self, value):
        self._verified = value

    @property
    def codegen_time(self):
        return self._codegen_time

    @codegen_time.setter
    def codegen_time(self, value):
        self._codegen_time = value

    @property
    def opt_time(self):
        return self._opt_time

    @opt_time.setter
    def opt_time(self, value):
        self._opt_time = value

    @property
    def executed(self):
        return self._executed

    @executed.setter
    def executed(self, value):
        self._executed = value

    @property
    def failed(self):
        return self._failed

    @failed.setter
    def failed(self, value):
        self._failed = value

    def hash(self):
        hasher = hashlib.sha256()
        hasher.update(str(self._specialize).encode("utf-8"))
        hasher.update(str(self._max_threads).encode("utf-8"))
        hasher.update(str(self._min_blocks_per_sm).encode("utf-8"))
        hasher.update(str(self._specialize_dims).encode("utf-8"))
        hasher.update(str(self._passes).encode("utf-8"))
        hasher.update(str(self._prune).encode("utf-8"))
        hasher.update(str(self._internalize).encode("utf-8"))
        hasher.update(str(self._codegen_opt).encode("utf-8"))
        hasher.update(str(self._rtc).encode("utf-8"))
        hasher.update(str(self._device_arch).encode("utf-8"))
        return hasher.hexdigest()

    @classmethod
    def from_dict(cls, **kwargs):
        return cls(**kwargs)

    def to_dict(self):
        data = {}
        data["specialize"] = self._specialize
        data["max_threads"] = self._max_threads
        data["min_blocks_per_sm"] = self._min_blocks_per_sm
        data["specialize_dims"] = self._specialize_dims
        data["passes"] = self._passes
        data["prune"] = self._prune
        data["internalize"] = self._internalize
        data["codegen_opt"] = self._codegen_opt
        data["rtc"] = self._rtc
        data["device_arch"] = self._device_arch
        data["failed"] = self.failed

        data["opt_time"] = self._opt_time
        data["codegen_time"] = self._codegen_time
        data["verified"] = self._verified
        data["obj_size"] = self._obj_size
        data["exec_time"] = self._exec_time
        data["executed"] = self._executed
        return data

    def dump(self):
        print(json.dumps(self.to_dict(), indent=6))

from sarathi.config import SchedulerType
from sarathi.core.scheduler.fcfs_disagg_emulation_scheduler import FCFSDisaggEmulationScheduler
from sarathi.core.scheduler.mlfq_disagg_emulation_scheduler import MLFQDisaggEmulationScheduler
from sarathi.core.scheduler.sarathi_scheduler import SarathiScheduler
from sarathi.utils.base_registry import BaseRegistry


class SchedulerRegistry(BaseRegistry):

    @classmethod
    def get_key_from_str(cls, key_str: str) -> SchedulerType:
        return SchedulerType.from_str(key_str)


SchedulerRegistry.register(SchedulerType.SARATHI, SarathiScheduler)
SchedulerRegistry.register(SchedulerType.FCFS_DISAGG_EMULATION, FCFSDisaggEmulationScheduler)
SchedulerRegistry.register(SchedulerType.MLFQ_DISAGG_EMULATION, MLFQDisaggEmulationScheduler)

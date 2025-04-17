from collections import deque
from enum import Enum, auto

from modules import MoistureAnalysis, RGBAnalysis, SatelliteLoader, LocationExtraction, ClimateReport, WaterPreprocessing, WaterAnalysis, WaterRGBAnalysis
from modules.module import ModuleResult


class DependencyResult(Enum):
    """Enum to represent the result of a dependency check."""
    ALL_SATISFIED = auto()
    MISSING = auto()
    STOP_PIPELINE = auto()


def main():
    modules = [LocationExtraction(),
               MoistureAnalysis(),
               ClimateReport(),
               RGBAnalysis(),
               WaterPreprocessing(),
               WaterAnalysis(),
               WaterRGBAnalysis(),
               SatelliteLoader()]

    # Sets to keep track of run and failed modules
    run_set = set()
    failed_set = set()

    # Dictionary for quick module lookup by name
    module_dict = {module.name: module for module in modules}

    # Stack to manage module processing order
    module_deque = deque(modules)

    def check_dependencies(module) -> DependencyResult:
        """Check if all dependencies and soft dependencies of the module have been run."""
        dependencies = module.dependencies
        soft_dependencies = module.soft_dependencies

        if any(dep in failed_set for dep in dependencies):
            print(f"Skipping {module.name} as one or more hard dependencies failed or requested a stop of their dependency pipeline.")
            return DependencyResult.STOP_PIPELINE
        elif not all(dep in run_set for dep in dependencies):
            missing_deps = [module_dict[dep] for dep in dependencies if dep not in run_set]
            module_deque.extend(missing_deps)
            return DependencyResult.MISSING
        elif not all(soft_dep in run_set or soft_dep in failed_set for soft_dep in soft_dependencies):
            missing_soft_deps = [module_dict[soft_dep] for soft_dep in soft_dependencies if soft_dep not in run_set and soft_dep not in failed_set]
            module_deque.extend(missing_soft_deps)
            return DependencyResult.MISSING
        return DependencyResult.ALL_SATISFIED

    def execute_module(module) -> None:
        """Execute the module and handle the result."""
        print(f"Running module {module.name}")
        match module.main():
            case ModuleResult.OK:
                print(f"Module {module.name} finished successfully")
                run_set.add(module.name)
            case ModuleResult.ERROR:
                print(f"Module {module.name} failed with an error")
                failed_set.add(module.name)
            case ModuleResult.STOP_PIPELINE:
                print(f"Module {module.name} requested to stop its pipeline")
                failed_set.add(module.name)

    while module_deque:
        module = module_deque.pop()

        if module.name in run_set or module.name in failed_set:
            continue

        match check_dependencies(module):
            case DependencyResult.ALL_SATISFIED:
                execute_module(module)
            case DependencyResult.MISSING:
                module_deque.appendleft(module)
            case DependencyResult.STOP_PIPELINE:
                failed_set.add(module.name)


if __name__ == "__main__":
    main()

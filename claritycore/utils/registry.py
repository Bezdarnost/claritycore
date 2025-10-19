# Copyright (c) Aman Urumbekov and other contributors.
class Registry:
    """
    A simple registry class that maps names to objects.
    It allows for dynamic instantiation of classes from a configuration file.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, module_name):
        """Decorator to register a new class."""

        def decorator(module_cls):
            if module_name in self._module_dict:
                raise KeyError(f"{module_name} is already registered in {self._name}!")
            self._module_dict[module_name] = module_cls
            return module_cls

        return decorator

    def get(self, name):
        """Retrieves a class by its registered name."""
        if name not in self._module_dict:
            raise KeyError(f"{name} is not registered in {self._name} registry")
        return self._module_dict[name]


# Create global instances for different components of your project
MODEL_REGISTRY = Registry("model")
LOSS_REGISTRY = Registry("loss")
METRIC_REGISTRY = Registry("metric")
ARCH_REGISTRY = Registry("arch")
DATASET_REGISTRY = Registry("dataset")
TRAINER_REGISTRY = Registry("trainer")

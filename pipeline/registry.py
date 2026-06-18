"""
registry.py
===========
Maps experiment_type -> {schema, validator, required fields}.

Adding a new experiment family = adding ONE plugin (a decorated validator in
pipeline/semantics/). The pipeline core and the perception layers never change.
Pages with no matching plugin, or too few fields to validate, DEGRADE
GRACEFULLY — perception output (Levels 1-3) still stands; we just skip domain
validation instead of crashing.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class Plugin:
    experiment_type: str
    schema: str               # path to this family's JSON schema
    required: tuple           # field keys needed to attempt domain validation
    validator: Callable       # (fields: dict) -> validation report


@dataclass
class DegradedResult:
    """Returned when no validator applies. validated=False is the signal."""
    experiment_type: str
    reason: str
    validated: bool = False

    def __str__(self):
        return (f"DEGRADED ({self.experiment_type}): {self.reason}\n"
                f"  -> perception output kept; domain validation skipped.")


_REGISTRY: dict[str, Plugin] = {}


def value_of(fields: dict, key: str, default=None):
    """Pull a scalar from the unified field envelope {value, confidence, ...}."""
    f = fields.get(key)
    return f.get("value", default) if isinstance(f, dict) else default


def plugin(experiment_type: str, schema: str, required):
    """Decorator: a validator self-registers as the plugin for a family."""
    def deco(fn: Callable) -> Callable:
        _REGISTRY[experiment_type] = Plugin(
            experiment_type, schema, tuple(required), fn)
        return fn
    return deco


def get(experiment_type: str) -> Optional[Plugin]:
    return _REGISTRY.get(experiment_type)


def registered() -> list:
    return sorted(_REGISTRY)


def dispatch(experiment_type: str, fields: dict):
    """Route to the matching validator, or degrade gracefully."""
    plug = _REGISTRY.get(experiment_type)
    if plug is None:
        return DegradedResult(
            experiment_type, f"no validator registered for '{experiment_type}'")
    missing = [k for k in plug.required if value_of(fields, k) is None]
    if missing:
        return DegradedResult(
            experiment_type, f"missing required fields {missing}")
    return plug.validator(fields)


if __name__ == "__main__":
    # demo: register a throwaway plugin and show the three dispatch outcomes
    @plugin("demo", schema="schema/demo.json", required=["a", "b"])
    def _demo_validator(fields):
        return f"validated demo: a+b = {value_of(fields,'a') + value_of(fields,'b')}"

    print("registered plugins:", registered())
    ok = {"a": {"value": 2}, "b": {"value": 3}}
    print(dispatch("demo", ok))
    print(dispatch("demo", {"a": {"value": 2}}))      # missing 'b' -> degrade
    print(dispatch("spectroscopy", ok))               # no plugin   -> degrade

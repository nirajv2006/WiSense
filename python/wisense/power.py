"""Decision layer: occupancy prediction -> recommended power-usage level.

A raw classifier output changes every second and is sometimes wrong. Acting
on it directly would make lights/fans flicker. The advisor therefore adds:

1. Confidence gate     if the model is unsure (< MIN_CONFIDENCE), keep the current level.
2. Asymmetric dwell    step UP after STEP_UP_AFTER_S of consistent evidence
                       (people should not wait in the dark), step DOWN only after
                       STEP_DOWN_AFTER_S (avoid switching off when someone sits still).
3. Explanation         every decision carries a human-readable reason.

The advisor only *recommends*. WiSense does not control any appliance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config


@dataclass
class Recommendation:
    level: str              # ECO / STANDARD / FULL
    share: float            # nominal fraction of full power for the zone
    actions: str
    reason: str
    changed: bool


@dataclass
class PowerAdvisor:
    level: str = "STANDARD"                 # safe default before any evidence
    step_up_s: float = config.STEP_UP_AFTER_S
    step_down_s: float = config.STEP_DOWN_AFTER_S
    min_confidence: float = config.MIN_CONFIDENCE
    candidate: str | None = None
    candidate_since: float | None = None
    history: list = field(default_factory=list)

    def update(self, occupancy: str, confidence: float, now: float) -> Recommendation:
        target = config.POWER_LEVELS[occupancy]["level"]
        order = config.POWER_ORDER

        if confidence < self.min_confidence:
            self.candidate = None
            return self._rec(f"Holding: model confidence {confidence:.0%} is below "
                             f"{self.min_confidence:.0%}", False)
        if target == self.level:
            self.candidate = None
            return self._rec(f"{occupancy.capitalize()} occupancy matches current level", False)

        if target != self.candidate:
            self.candidate, self.candidate_since = target, now
        going_up = order.index(target) > order.index(self.level)
        need = self.step_up_s if going_up else self.step_down_s
        waited = now - self.candidate_since
        if waited >= need:
            old, self.level, self.candidate = self.level, target, None
            self.history.append((now, old, target))
            return self._rec(f"{'Increased' if going_up else 'Reduced'} from {old}: "
                             f"{occupancy} occupancy for {waited:.0f} s", True)
        return self._rec(f"{occupancy.capitalize()} occupancy detected - "
                         f"{'raising' if going_up else 'reducing'} to {target} in {need - waited:.0f} s "
                         f"if it persists", False)

    def _rec(self, reason: str, changed: bool) -> Recommendation:
        info = next(v for v in config.POWER_LEVELS.values() if v["level"] == self.level)
        return Recommendation(self.level, info["share"], info["actions"], reason, changed)

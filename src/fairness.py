"""Fairness utilities.

Ethical Motivation
------------------
Predictive policing systems risk creating feedback loops where certain communities
are repeatedly over-policed, leading to:
  - Increased surveillance and arrests in already marginalized areas
  - Self-fulfilling prophecies (more patrols → more arrests → higher predicted risk)
  - Erosion of community trust and civil liberties

This module implements a fairness constraint to prevent repeated consecutive
assignments to the same zone, promoting more equitable distribution of patrol
resources across the city.

The constraint can be overridden for genuinely high-risk situations, balancing
fairness with public safety needs.
"""

from typing import Any


def is_allowed(
    zone: int,
    state: dict[str, Any],
    risk_score: float,
    cfg: dict[str, Any]
) -> bool:
    """
    Check if assigning a patrol to a zone is allowed under fairness constraints.

    Fairness Rule
    -------------
    A zone cannot be assigned patrols for more than `max_consecutive` times
    in a row, UNLESS the predicted risk score exceeds `override_risk_threshold`.

    This prevents over-policing of specific communities while allowing emergency
    response to genuinely high-risk situations.

    Args:
        zone: Community area ID
        state: Fairness state dict with:
            - "consecutive_count": dict mapping zone -> consecutive assignment count
            - "violations": total number of fairness violations (for auditing)
        risk_score: Predicted risk score for this zone (normalized 0-1 or class probability)
        cfg: Configuration dict with:
            - "max_consecutive": maximum consecutive assignments allowed (default: 2)
            - "override_risk_threshold": risk threshold to override fairness (default: 0.75)

    Returns:
        True if assignment is allowed, False if blocked by fairness constraint

    Side Effects:
        Increments state["violations"] if assignment is blocked
    """
    # Get configuration
    max_consecutive = cfg.get("max_consecutive", 2)
    override_threshold = cfg.get("override_risk_threshold", 0.75)

    # Get current consecutive count for this zone
    consecutive_count = state.get("consecutive_count", {})
    current_count = consecutive_count.get(zone, 0)

    # Check if we've exceeded the limit
    if current_count >= max_consecutive:
        # Allow override if risk is genuinely high
        if risk_score >= override_threshold:
            return True
        else:
            # Block assignment and record violation
            if "violations" not in state:
                state["violations"] = 0
            state["violations"] += 1
            return False

    return True


def update_state(
    assigned_zones: list[int],
    state: dict[str, Any]
) -> None:
    """
    Update fairness state after patrol assignments.

    Ethical Consideration
    ---------------------
    This function implements the "memory" of the fairness system. By tracking
    consecutive assignments and resetting counts for unassigned zones, we ensure
    that patrol distribution rotates across different communities rather than
    concentrating on the same areas repeatedly.

    Args:
        assigned_zones: List of zone IDs that received patrol assignments
        state: Fairness state dict (modified in-place) with:
            - "consecutive_count": dict mapping zone -> consecutive assignment count

    Side Effects:
        - Increments consecutive_count for assigned zones
        - Resets consecutive_count to 0 for zones not assigned
    """
    # Initialize consecutive_count if not present
    if "consecutive_count" not in state:
        state["consecutive_count"] = {}

    consecutive_count = state["consecutive_count"]

    # Get all zones we've seen before
    all_zones = set(consecutive_count.keys())
    assigned_set = set(assigned_zones)

    # Increment count for assigned zones
    for zone in assigned_zones:
        consecutive_count[zone] = consecutive_count.get(zone, 0) + 1

    # Reset count for zones not assigned
    for zone in all_zones - assigned_set:
        consecutive_count[zone] = 0


def initialize_state() -> dict[str, Any]:
    """
    Initialize a new fairness state.

    Returns:
        Empty fairness state dict with:
            - "consecutive_count": empty dict
            - "violations": 0
    """
    return {
        "consecutive_count": {},
        "violations": 0
    }


def get_fairness_report(state: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a fairness audit report from current state.

    Ethical Accountability
    ----------------------
    Transparency is crucial for algorithmic fairness. This function provides
    metrics that can be used to audit the system and identify potential biases
    or over-policing patterns.

    Args:
        state: Fairness state dict

    Returns:
        Report dict with:
            - "total_violations": number of times fairness constraint was triggered
            - "max_consecutive": highest consecutive count across all zones
            - "zones_at_limit": list of zones currently at their consecutive limit
            - "distribution": full consecutive_count dict for detailed analysis
    """
    consecutive_count = state.get("consecutive_count", {})
    violations = state.get("violations", 0)

    max_consecutive = max(consecutive_count.values()) if consecutive_count else 0

    # Find zones currently at high consecutive counts (potential concern)
    zones_at_limit = [
        zone for zone, count in consecutive_count.items()
        if count >= 2  # Using 2 as a warning threshold
    ]

    return {
        "total_violations": violations,
        "max_consecutive": max_consecutive,
        "zones_at_limit": zones_at_limit,
        "distribution": dict(consecutive_count)
    }


if __name__ == "__main__":
    # Demonstration of fairness system
    print("Fairness System Demonstration")
    print("=" * 60)

    # Initialize state and config
    state = initialize_state()
    cfg = {
        "max_consecutive": 2,
        "override_risk_threshold": 0.75
    }

    print("\nScenario 1: Normal assignment (first time)")
    zone_a = 10
    risk_low = 0.3
    allowed = is_allowed(zone_a, state, risk_low, cfg)
    print(f"  Zone {zone_a}, risk={risk_low:.2f}: {'ALLOWED' if allowed else 'BLOCKED'}")
    if allowed:
        update_state([zone_a], state)

    print("\nScenario 2: Second consecutive assignment")
    allowed = is_allowed(zone_a, state, risk_low, cfg)
    print(f"  Zone {zone_a}, risk={risk_low:.2f}: {'ALLOWED' if allowed else 'BLOCKED'}")
    if allowed:
        update_state([zone_a], state)

    print("\nScenario 3: Third consecutive (should block)")
    allowed = is_allowed(zone_a, state, risk_low, cfg)
    print(f"  Zone {zone_a}, risk={risk_low:.2f}: {'ALLOWED' if allowed else 'BLOCKED'}")

    print("\nScenario 4: Third consecutive with HIGH risk (override)")
    risk_high = 0.85
    allowed = is_allowed(zone_a, state, risk_high, cfg)
    print(f"  Zone {zone_a}, risk={risk_high:.2f}: {'ALLOWED' if allowed else 'BLOCKED'}")

    print("\nScenario 5: Assign to different zone (resets zone_a)")
    zone_b = 20
    update_state([zone_b], state)
    print(f"  Assigned to zone {zone_b}, zone {zone_a} count reset")

    print("\nScenario 6: Zone A allowed again after reset")
    allowed = is_allowed(zone_a, state, risk_low, cfg)
    print(f"  Zone {zone_a}, risk={risk_low:.2f}: {'ALLOWED' if allowed else 'BLOCKED'}")

    print("\n" + "=" * 60)
    print("Fairness Audit Report:")
    report = get_fairness_report(state)
    for key, value in report.items():
        print(f"  {key}: {value}")

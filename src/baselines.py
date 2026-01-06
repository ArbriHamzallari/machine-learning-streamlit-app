"""Baseline strategies for patrol allocation.

These baselines provide comparison points to evaluate the performance
of the constrained greedy planner.
"""

import random
from typing import Any

from src.planner import manhattan_distance


def allocate_random(
    zones: list[int],
    unit_positions: dict[str, tuple[int, int]],
    zone_xy: dict[int, tuple[int, int]],
    seed: int = 42
) -> tuple[dict[str, int], int]:
    """
    Random baseline: randomly assign each patrol unit to a zone.

    This baseline provides a lower bound on performance - any reasonable
    strategy should outperform random assignment.

    Args:
        zones: List of available zone IDs
        unit_positions: Dict mapping unit_id -> (x, y) current position
        zone_xy: Dict mapping zone_id -> (x, y) grid coordinates
        seed: Random seed for reproducibility

    Returns:
        assignments: Dict mapping unit_id -> assigned zone_id
        total_distance: Sum of Manhattan distances traveled by all units
    """
    random.seed(seed)

    assignments = {}
    total_distance = 0

    # Get list of units
    units = list(unit_positions.keys())

    # Randomly shuffle zones
    available_zones = zones.copy()
    random.shuffle(available_zones)

    # Assign one zone per unit (or fewer if not enough zones)
    for i, unit_id in enumerate(units):
        if i >= len(available_zones):
            # More units than zones - skip remaining units
            break

        assigned_zone = available_zones[i]
        assignments[unit_id] = assigned_zone

        # Calculate distance
        current_pos = unit_positions[unit_id]
        target_pos = zone_xy[assigned_zone]
        distance = manhattan_distance(current_pos, target_pos)
        total_distance += distance

        # Update unit position
        unit_positions[unit_id] = target_pos

    return assignments, total_distance


def allocate_risk_greedy(
    zones: list[int],
    risk_scores: dict[int, float],
    unit_positions: dict[str, tuple[int, int]],
    zone_xy: dict[int, tuple[int, int]]
) -> tuple[dict[str, int], int]:
    """
    Risk-greedy baseline: assign patrols to highest-risk zones only.

    This baseline ignores:
    - Distance considerations
    - Fairness constraints

    It represents a pure "go where the risk is highest" strategy, which
    may lead to over-policing of certain communities.

    Args:
        zones: List of available zone IDs
        risk_scores: Dict mapping zone_id -> risk score
        unit_positions: Dict mapping unit_id -> (x, y) current position
        zone_xy: Dict mapping zone_id -> (x, y) grid coordinates

    Returns:
        assignments: Dict mapping unit_id -> assigned zone_id
        total_distance: Sum of Manhattan distances traveled by all units
    """
    assignments = {}
    total_distance = 0

    # Sort zones by risk (descending)
    sorted_zones = sorted(
        zones,
        key=lambda z: risk_scores.get(z, 0.0),
        reverse=True
    )

    # Get list of units
    units = list(unit_positions.keys())

    # Assign highest-risk zones to units
    for i, unit_id in enumerate(units):
        if i >= len(sorted_zones):
            # More units than zones - skip remaining units
            break

        assigned_zone = sorted_zones[i]
        assignments[unit_id] = assigned_zone

        # Calculate distance
        current_pos = unit_positions[unit_id]
        target_pos = zone_xy[assigned_zone]
        distance = manhattan_distance(current_pos, target_pos)
        total_distance += distance

        # Update unit position
        unit_positions[unit_id] = target_pos

    return assignments, total_distance


if __name__ == "__main__":
    # Demonstration of baseline strategies
    print("Baseline Strategies Demonstration")
    print("=" * 60)

    # Setup test scenario
    zones = [1, 2, 3, 4, 5]
    zone_xy = {
        1: (0, 0),
        2: (2, 2),
        3: (5, 5),
        4: (7, 3),
        5: (9, 9)
    }
    risk_scores = {
        1: 0.2,
        2: 0.8,  # High risk
        3: 0.5,
        4: 0.9,  # Highest risk
        5: 0.3
    }

    # Test 1: Random allocation
    print("\n[1] Random Allocation")
    print("-" * 60)
    unit_positions_1 = {
        "unit_1": (0, 0),
        "unit_2": (5, 5)
    }
    assignments_1, distance_1 = allocate_random(
        zones, unit_positions_1.copy(), zone_xy, seed=42
    )
    print(f"Assignments: {assignments_1}")
    print(f"Total distance: {distance_1}")
    print(f"Assigned zones: {sorted(assignments_1.values())}")

    # Test 2: Risk-greedy allocation
    print("\n[2] Risk-Greedy Allocation")
    print("-" * 60)
    unit_positions_2 = {
        "unit_1": (0, 0),
        "unit_2": (5, 5)
    }
    assignments_2, distance_2 = allocate_risk_greedy(
        zones, risk_scores, unit_positions_2.copy(), zone_xy
    )
    print(f"Assignments: {assignments_2}")
    print(f"Total distance: {distance_2}")
    print(f"Assigned zones (by risk): {sorted(assignments_2.values())}")

    # Show risk scores for comparison
    print("\n[3] Risk Scores (for reference)")
    print("-" * 60)
    for zone in sorted(zones):
        print(f"  Zone {zone}: risk = {risk_scores[zone]:.2f}")

    print("\n" + "=" * 60)
    print("Comparison:")
    print(f"  Random distance: {distance_1}")
    print(f"  Risk-greedy distance: {distance_2}")
    print(f"  Risk-greedy targets highest-risk zones: {sorted(assignments_2.values())}")

from enum_class import *
from measurement import *
from shapely.geometry import Point, Polygon, LineString
import shapely.ops
import numpy as np

def attack_range_s2s(attacker_state : tuple[str, float, float, float], defender_state : tuple[str, float, float, float], extension_factor=500) -> tuple[dict[HullSection, bool],dict[HullSection, dict[HullSection, AttackRange]]]:
    """
    return:
        target_dict : dict[HullSection, bool] : whether each hull section can target any hull section
        measure_dict : dict[HullSection, dict[HullSection, AttackRange]] : the attack_range (AttackRange) for each hull section to each hull section
    """
    target_dict : dict[HullSection, bool] = {from_hull : False for from_hull in HullSection}
    measure_dict : dict[HullSection, dict[HullSection, AttackRange]] = {from_hull : {to_hull : AttackRange.INVALID for to_hull in HullSection} for from_hull in HullSection}
    
    attacker_coords = _ship_coordinate(attacker_state)
    defender_coords = _ship_coordinate(defender_state)

    attacker_center : np.ndarray = attacker_coords['center_point']
    defender_center : np.ndarray = defender_coords['center_point']
    # orientation vector points to the front of the ship
    attacker_orientation_vector : np.ndarray = np.array([np.sin(attacker_state[3]), np.cos(attacker_state[3])])
    

    # # distance check
    # target_vector : np.ndarray = attacker_center - defender_center
    # distance = np.linalg.norm(target_vector)
    # if distance > 2 * LONG_RANGE :
    #     return target_dict, measure_dict
    

    
    attacker_poly : dict[HullSection, Polygon] = create_hull_poly(attacker_coords['arc_points'])
    defender_poly : dict[HullSection, Polygon] = create_hull_poly(defender_coords['arc_points'])

    ROTATION_MATRICES = [np.linalg.matrix_power(np.array([[0, 1], [-1, 0]]), i) for i in range(4)]

    for from_hull in HullSection :
        for to_hull in HullSection :
            from_hull_targeting_pt = attacker_coords['targeting_points'][from_hull]
            to_hull_targeting_pt = defender_coords['targeting_points'][to_hull]

            # attack hull orientation check
            attack_orientation_vector = ROTATION_MATRICES[from_hull] @ attacker_orientation_vector
            hull_target_vector = to_hull_targeting_pt - from_hull_targeting_pt
            dot_product = np.dot(attack_orientation_vector, hull_target_vector)
            if dot_product < 0 :
                continue

            # Line of Sight blocked
            is_blocked : bool = False
            line_of_sight = LineString([from_hull_targeting_pt, to_hull_targeting_pt])
            for hull in HullSection:
                if hull != to_hull and line_of_sight.crosses(defender_poly[hull].exterior):
                    is_blocked = True
                    continue
            if is_blocked : 
                continue

            # Range
            from_hull_poly = attacker_poly[from_hull]
            to_hull_poly = defender_poly[to_hull]

            if from_hull in (HullSection.FRONT, HullSection.RIGHT) :
                arc1_center, arc1_end = attacker_coords['arc_points'][0], attacker_coords['arc_points'][2]
            else :
                arc1_center, arc1_end = attacker_coords['arc_points'][3], attacker_coords['arc_points'][4]

            if from_hull in (HullSection.FRONT, HullSection.LEFT) :
                arc2_center, arc2_end = attacker_coords['arc_points'][0], attacker_coords['arc_points'][1]
            else :
                arc2_center, arc2_end = attacker_coords['arc_points'][3], attacker_coords['arc_points'][5]

            # Build the arc polygon. This logic works for ALL hull sections.
            vec1 = np.array(arc1_end) - np.array(arc1_center)
            vec2 = np.array(arc2_end) - np.array(arc2_center) # Note: vector points away from the ship
            arc_polygon = Polygon([
                arc1_end,
                np.array(arc1_end) + vec1 * extension_factor,
                np.array(arc2_end) + vec2 * extension_factor,
                arc2_end
            ])

            target_hull = to_hull_poly.exterior
            to_hull_in_arc = target_hull.intersection(arc_polygon)

            if to_hull_in_arc.is_empty :
                continue # not in arc

            range_measure = LineString(shapely.ops.nearest_points(from_hull_poly.exterior, to_hull_in_arc))
            is_blocked = False
            for hull in HullSection :
                if hull != to_hull and range_measure.crosses(defender_poly[hull].exterior) :
                    is_blocked = True
                    break
            if is_blocked : continue

            distance = range_measure.length

            if distance <= CLOSE_RANGE : 
                measure_dict[from_hull][to_hull] = AttackRange.CLOSE 
                target_dict[from_hull] = True
            elif distance <= MEDIUM_RANGE : 
                measure_dict[from_hull][to_hull] = AttackRange.MEDIUM
                target_dict[from_hull] = True
            elif distance <= LONG_RANGE : 
                measure_dict[from_hull][to_hull] = AttackRange.LONG
                target_dict[from_hull] = True
            else : measure_dict[from_hull][to_hull] = AttackRange.EXTREME 
    from test1 import attack_range_s2s_numba
    numba_resault = attack_range_s2s_numba(attacker_state, defender_state, extension_factor=500)
    if (target_dict, measure_dict) != numba_resault :
        raise ValueError(f"Numba and non-Numba results do not match!\n{attacker_state} to {defender_state}\n{measure_dict}\n{numba_resault[1]}")
    return target_dict, measure_dict

def attack_range_s2q(ship_state : tuple[str, float, float, float], squad_state : tuple[float, float], extension_factor=500) -> dict[HullSection, AttackRange]:
    """
    return:
        attack_range (AttackRange) for each hull section
    """
    range_dict = {hull : AttackRange.INVALID for hull in HullSection}

    ship_coords = _ship_coordinate(ship_state)
    squad_coords = np.array(squad_state)
    attacker_orientation_vector : np.ndarray = np.array([np.sin(ship_state[3]), np.cos(ship_state[3])])
    attacker_poly : dict[HullSection, Polygon] = create_hull_poly(ship_coords['arc_points'])

    ROTATION_MATRICES = [np.linalg.matrix_power(np.array([[0, 1], [-1, 0]]), i) for i in range(4)]

    for ship_hull in HullSection :
        from_hull_targeting_pt = ship_coords['targeting_points'][ship_hull]
        squad_targeting_pt = squad_coords

        # attack hull orientation check
        attack_orientation_vector = ROTATION_MATRICES[ship_hull] @ attacker_orientation_vector
        hull_target_vector = squad_targeting_pt - from_hull_targeting_pt
        dot_product = np.dot(attack_orientation_vector, hull_target_vector)
        if dot_product < 0 :
            continue

        # Range
        hull_poly = attacker_poly[ship_hull]
        squad_token = Point(squad_targeting_pt).buffer(SQUAD_TOKEN_RADIUS)

        if ship_hull in (HullSection.FRONT, HullSection.RIGHT) :
            arc1_center, arc1_end = ship_coords['arc_points'][0], ship_coords['arc_points'][2]
        else :
            arc1_center, arc1_end = ship_coords['arc_points'][3], ship_coords['arc_points'][4]

        if ship_hull in (HullSection.FRONT, HullSection.LEFT) :
            arc2_center, arc2_end = ship_coords['arc_points'][0], ship_coords['arc_points'][1]
        else :
            arc2_center, arc2_end = ship_coords['arc_points'][3], ship_coords['arc_points'][5]

        # Build the arc polygon. This logic works for ALL hull sections.
        vec1 = np.array(arc1_end) - np.array(arc1_center)
        vec2 = np.array(arc2_end) - np.array(arc2_center) # Note: vector points away from the ship
        arc_polygon = Polygon([
            arc1_end,
            np.array(arc1_end) + vec1 * extension_factor,
            np.array(arc2_end) + vec2 * extension_factor,
            arc2_end
        ])

        to_hull_in_arc = squad_token.intersection(arc_polygon)

        if to_hull_in_arc.is_empty :
            continue # not in arc

        distance = LineString(shapely.ops.nearest_points(hull_poly.exterior, to_hull_in_arc)).length

        if distance <= CLOSE_RANGE : 
            range_dict[ship_hull]= AttackRange.CLOSE
        elif distance <= MEDIUM_RANGE : 
            range_dict[ship_hull]= AttackRange.MEDIUM 
        elif distance <= LONG_RANGE : 
            range_dict[ship_hull]= AttackRange.LONG 
        else : range_dict[ship_hull]= AttackRange.EXTREME 

    return range_dict

def attack_range_q2s(squad_state : tuple[float, float], ship_state : tuple[str, float, float, float]) -> dict[HullSection, bool]:
    """
    return:
        in_range (bool) for each hull section
    """
    in_range_dict = {hull : False for hull in HullSection}

    ship_coords = _ship_coordinate(ship_state)
    squad_coords = np.array(squad_state)
    ship_poly : tuple[np.ndarray,...] = create_hull_arrays(ship_coords['arc_points'])
    squad_token = Point(squad_coords).buffer(SQUAD_TOKEN_RADIUS)

    
    for target_hull in HullSection :
        is_blocked : bool = False
        hull_poly = ship_poly[target_hull]
        attack_range = LineString(shapely.ops.nearest_points(hull_poly.exterior, squad_token))
        for hull in HullSection :
            if hull != target_hull and attack_range.crosses(ship_poly[hull].exterior) :
                is_blocked = True
                break
            if is_blocked : continue

        if attack_range.length <= DISTANCE[1] + SQUAD_TOKEN_RADIUS: 
            in_range_dict[target_hull]= True

    return in_range_dict

    
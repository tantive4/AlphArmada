from PIL import Image, ImageDraw, ImageFont
import ship as ship_module

def visualize(game, title : str, maneuver_tool = None) -> None:
    """Creates and saves an image of the current game state with (0,0) at the bottom-left."""
    img = Image.new('RGB', (game.player_edge, game.short_edge), 'black')
    draw = ImageDraw.Draw(img)

    # Helper function to transform a coordinate from game space (0,0 at bottom-left)
    # to image space (0,0 at top-left).
    def transform_coord(coord):
        return (coord[0], game.short_edge - coord[1])

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Keep the title at the top-left of the image for readability
    display_title = f"Round {game.round} | {title}"
    draw.text((10, 10), display_title, font=font, fill='white')

    for ship in game.ships:
        if ship.destroyed:
            continue

        # Transform and draw the ship's base
        base_coords = [transform_coord(p) for p in [ship.front_left_base, ship.front_right_base, ship.rear_right_base, ship.rear_left_base]]
        draw.polygon(base_coords, outline='white')

        # Transform and draw the firing arcs
        draw.line([transform_coord(ship.front_arc_center), transform_coord(ship.front_left_arc)], fill='yellow')
        draw.line([transform_coord(ship.front_arc_center), transform_coord(ship.front_right_arc)], fill='yellow')
        draw.line([transform_coord(ship.rear_arc_center), transform_coord(ship.rear_left_arc)], fill='red')
        draw.line([transform_coord(ship.rear_arc_center), transform_coord(ship.rear_right_arc)], fill='red')

        # --- Text Labels ---
        # Positions are defined in game coordinates (y-up) and then transformed.

        # Ship Name (Positioned above the ship's center)
        name_pos = (ship.x - 20, ship.y + 40)
        draw.text(transform_coord(name_pos), ship.name, font=font, fill='cyan')
        
        # Hull (Positioned near the ship's center)
        hull_pos = (ship.x - 5, ship.y + 10)
        draw.text(transform_coord(hull_pos), str(ship.hull), font=font, fill='green')
        
        # Shields
        # Front (Positioned "above" the front edge)
        front_shield_pos = (
            (ship.front_left_base[0] + ship.front_right_base[0]) / 2, 
            (ship.front_left_base[1] + ship.front_right_base[1]) / 2 + 15
        )
        draw.text(transform_coord(front_shield_pos), str(ship.shield[0]), font=font, fill='blue')
        
        # Right (Positioned to the right of the side edge)
        right_shield_pos = (
            (ship.front_right_base[0] + ship.rear_right_base[0]) / 2 + 5, 
            (ship.front_right_base[1] + ship.rear_right_base[1]) / 2
        )
        draw.text(transform_coord(right_shield_pos), str(ship.shield[1]), font=font, fill='blue')

        # Rear (Positioned "below" the rear edge)
        rear_shield_pos = (
            (ship.rear_left_base[0] + ship.rear_right_base[0]) / 2, 
            (ship.rear_left_base[1] + ship.rear_right_base[1]) / 2 - 15
        )
        draw.text(transform_coord(rear_shield_pos), str(ship.shield[2]), font=font, fill='blue')
        
        # Left (Positioned to the left of the side edge)
        left_shield_pos = (
            (ship.front_left_base[0] + ship.rear_left_base[0]) / 2 - 15, 
            (ship.front_left_base[1] + ship.rear_left_base[1]) / 2
        )
        draw.text(transform_coord(left_shield_pos), str(ship.shield[3]), font=font, fill='blue')
    
    if maneuver_tool:
        # Draw the maneuver tool path
        transformed_tool_path = [transform_coord(p) for p in maneuver_tool]
        draw.line(transformed_tool_path, fill='grey', width=int(ship_module.TOOL_WIDTH))
        
        # Draw the joints
        radius = ship_module.TOOL_WIDTH / 2
        for i, point in enumerate(maneuver_tool):
            if i % 2 != 0:
                transformed_point = transform_coord(point)
                # Define the bounding box for the circle
                bounding_box = [
                    (transformed_point[0] - radius, transformed_point[1] - radius),
                    (transformed_point[0] + radius, transformed_point[1] + radius)
                ]
                draw.ellipse(bounding_box, fill='darkgrey')


    img.save(f'game_visualizer/game_state_{game.image_counter}.png')
    game.image_counter += 1
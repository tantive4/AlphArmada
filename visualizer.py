from PIL import Image, ImageDraw, ImageFont
import ship as ship_module
from typing import TYPE_CHECKING
# Conditionally import Armada only for type checking
if TYPE_CHECKING:
    from armada import Armada
    
def visualize(game : "Armada", title : str,  maneuver_tool : list[tuple[float, float]] | None = None) -> None:
    """Creates and saves an image of the current game state with (0,0) at the bottom-left."""
    img = Image.new('RGB', (game.player_edge, game.short_edge), (0,0,0)) # type: ignore
    draw = ImageDraw.Draw(img)

    # Helper function to transform a coordinate from game space (0,0 at bottom-left)
    # to image space (0,0 at top-left).
    def transform_coord(coord):
        return (coord[0], game.short_edge - coord[1])

    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()

    # Keep the title at the top-left of the image for readability
    display_title = title
    draw.text((10, 10), display_title, font=font, fill='white')

    for ship in game.ships:
        if ship.destroyed:
            continue

        # Transform and draw the ship's base
        base_coords = [transform_coord(p) for p in [ship.front_left_base, ship.front_right_base, ship.rear_right_base, ship.rear_left_base]]
        draw.polygon(base_coords, outline='white')

        # Transform and draw the firing arcs
        draw.line([transform_coord(ship.front_arc_center), transform_coord(ship.front_left_arc)], fill='red')
        draw.line([transform_coord(ship.front_arc_center), transform_coord(ship.front_right_arc)], fill='red')
        draw.line([transform_coord(ship.rear_arc_center), transform_coord(ship.rear_left_arc)], fill='red')
        draw.line([transform_coord(ship.rear_arc_center), transform_coord(ship.rear_right_arc)], fill='red')

        # Transform and draw the targeting points
        for point in ship.targeting_point.values():
            p_transformed = transform_coord(point)
            dot_size = 1
            bounding_box = [
                (p_transformed[0] - dot_size, p_transformed[1] - dot_size),
                (p_transformed[0] + dot_size, p_transformed[1] + dot_size)
            ]
            draw.ellipse(bounding_box, fill='yellow', outline='yellow')

        # --- Text Labels ---
        # Positions are defined in game coordinates (y-up) and then transformed.

        # Ship Name (Positioned above the ship's center, offset for clarity)
        name_pos = ship._get_coordination(-15, 25)
        draw.text(transform_coord(name_pos), ship.name, font=font, fill='cyan')
        
        # Hull (Positioned near the ship's center)
        hull_pos = ship._get_coordination(0, -15)
        draw.text(transform_coord(hull_pos), str(ship.hull), font=font, fill='green')
        
        # Shields
        # Front (Positioned "above" the front edge)
        front_shield_pos = (
            (ship.front_left_base[0] + ship.front_right_base[0]) / 2, 
            (ship.front_left_base[1] + ship.front_right_base[1]) / 2
        )
        draw.text(transform_coord(front_shield_pos), str(ship.shield[ship_module.HullSection.FRONT]), font=font, fill='cyan')
        
        # Right (Positioned to the right of the side edge)
        right_shield_pos = (
            (ship.front_right_base[0] + ship.rear_right_base[0]) / 2, 
            (ship.front_right_base[1] + ship.rear_right_base[1]) / 2
        )
        draw.text(transform_coord(right_shield_pos), str(ship.shield[ship_module.HullSection.RIGHT]), font=font, fill='cyan')

        # Rear (Positioned "below" the rear edge)
        rear_shield_pos = (
            (ship.rear_left_base[0] + ship.rear_right_base[0]) / 2, 
            (ship.rear_left_base[1] + ship.rear_right_base[1]) / 2
        )
        draw.text(transform_coord(rear_shield_pos), str(ship.shield[ship_module.HullSection.REAR]), font=font, fill='cyan')
        
        # Left (Positioned to the left of the side edge)
        left_shield_pos = (
            (ship.front_left_base[0] + ship.rear_left_base[0]) / 2, 
            (ship.front_left_base[1] + ship.rear_left_base[1]) / 2
        )
        draw.text(transform_coord(left_shield_pos), str(ship.shield[ship_module.HullSection.LEFT]), font=font, fill='cyan')
        
        # Defense Tokens
        displayed_tokens = [t for t in ship.defense_tokens if not t.discarded]
        if displayed_tokens:
            first_token = displayed_tokens[0]
            bbox = font.getbbox(first_token.type.name)
            text_width = bbox[2] - bbox[0]
            
            x_offset = -text_width / 2
            y_offset = -ship.base_size[1] * 1.3
            start_pos_game = ship._get_coordination(x_offset, y_offset)
            start_x_img, start_y_img = transform_coord(start_pos_game)

            for i, token in enumerate(displayed_tokens):
                token_text = token.type.name
                bbox = font.getbbox(token_text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                y_img = start_y_img + i * (text_height + 10)
                x_img = start_x_img
                token_pos_img = (x_img, y_img)
                
                bg_color = 'green' if token.readied else 'red'
                
                # Define background rectangle directly in image coordinates
                bg_coords = [x_img - 2, y_img - 2, x_img + text_width + 2, y_img + text_height + 2]

                draw.rectangle(bg_coords, fill=bg_color)
                draw.text(token_pos_img, token_text, font=font, fill='white')


        # Command Tokens
        if ship.command_token:
            first_command = ship.command_token[0]
            bbox = font.getbbox(str(first_command))
            text_width = bbox[2] - bbox[0]
            
            x_offset = ship.base_size[0] * 1.2
            y_offset = -ship.base_size[1] * 1.2
            start_pos_game = ship._get_coordination(x_offset, y_offset)
            start_x_img, start_y_img = transform_coord(start_pos_game)
            
            for i, command in enumerate(ship.command_token):
                token_text = str(command)
                bbox = font.getbbox(token_text)
                text_height = bbox[3] - bbox[1]
                
                y_img = start_y_img + i * (text_height + 10)
                x_img = start_x_img
                token_pos_img = (x_img, y_img)

                draw.text(token_pos_img, token_text, font=font, fill='white')


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
    import os

    # Create a directory to store visuals if it doesn't exist
    output_dir = "game_visuals"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the image inside that directory
    img.save(os.path.join(output_dir, f'game_state_{game.image_counter}.png'))
    game.image_counter += 1
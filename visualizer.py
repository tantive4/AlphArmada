from PIL import Image, ImageDraw, ImageFont

import math
from typing import TYPE_CHECKING
import os

import ship as ship_module

# Conditionally import Armada only for type checking
if TYPE_CHECKING:
    from armada import Armada
    
def _draw_ship_template(ship: ship_module.Ship, font: ImageFont.FreeTypeFont) -> Image.Image:
    """
    Creates an image of a single ship with all its details on a transparent background.
    The ship is drawn with its front-token-center pivot at the center of the template image.
    """
    # Use a fixed-size canvas for all templates
    width, height = 400, 400
    template_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(template_img)
    
    # The "ZERO POINT" is the center of the template, corresponding to the ship's pivot
    origin_x, origin_y = width / 2, height / 2

    # Function to translate ship-local coordinates (y-up) to template image coordinates (y-down)
    def to_template_coord(p):
        """
        Takes offset from **Front Center of Ship Token** as input\n
        +y side is front side of ship
        """
        return (p[0] + origin_x, -p[1] + origin_y)

    # Draw the ship's base and hull zones using the template vertices
    base_coords = [to_template_coord(p) for p in ship.template_base_vertices]
    draw.polygon(base_coords, outline='white')
    for hull_vertices in ship.template_hull_vertices.values():
        hull_coords = [to_template_coord(p) for p in hull_vertices]
        draw.polygon(hull_coords, outline='red')

    # Draw targeting points
    for hull in ship_module.HullSection:
        point = ship.template_targeting_points_and_maneuver_tool_insert[hull.value]
        p_transformed = to_template_coord(point)
        dot_size = 1
        bounding_box = [
            (p_transformed[0] - dot_size, p_transformed[1] - dot_size),
            (p_transformed[0] + dot_size, p_transformed[1] + dot_size)
        ]
        draw.ellipse(bounding_box, fill='yellow', outline='yellow')

    # --- Draw Text Labels (relative to the pivot) ---
    draw.text(to_template_coord((0, 15)), ship.name, font=font, fill='white', anchor="ms")
    draw.text(to_template_coord((0, -24)), str(ship.hull), font=font, fill='yellow', anchor="mm")
    
    # Shields
    draw.text(to_template_coord((0, 2)), str(ship.shield[ship_module.HullSection.FRONT]), font=font, fill='cyan', anchor="mb")
    draw.text(to_template_coord((0, -ship.token_size[1] - 12)), str(ship.shield[ship_module.HullSection.REAR]), font=font, fill='cyan', anchor="ms")
    draw.text(to_template_coord((ship.token_size[0]/2 + 5, -ship.token_size[1]/2)), str(ship.shield[ship_module.HullSection.RIGHT]), font=font, fill='cyan', anchor="lm")
    draw.text(to_template_coord((-ship.token_size[0]/2 - 5, -ship.token_size[1]/2)), str(ship.shield[ship_module.HullSection.LEFT]), font=font, fill='cyan', anchor="rm")
    
    draw.text(to_template_coord((ship.token_size[0]/2 - 10, -ship.token_size[1] + 5)), str(ship.speed), font=font,fill='white', anchor='ls')

    # --- Draw Defense Tokens ---
    start_x, start_y = to_template_coord((ship.base_size[0] / 2 + 5, -ship.base_size[1] - 5))
    valid_tokens_with_keys = [
        (key, token) for key, token in ship.defense_tokens.items() if not token.discarded
    ]
    sorted_tokens = sorted(valid_tokens_with_keys, key=lambda item: item[0])
    # Iterate backwards to stack from the bottom up
    for i, (key, token) in enumerate(reversed(sorted_tokens)):
        token_text = token.type.name
        # Stack upwards from the starting position
        pos_y = start_y - i * 12
        bg_color = 'green' if token.readied else 'red'
        
        bbox = draw.textbbox((start_x, pos_y), token_text, font=font, anchor="ls")
        draw.rectangle(bbox, fill=bg_color)
        draw.text((start_x, pos_y), token_text, font=font, fill='white', anchor="ls")

    # --- Draw Command Tokens ---
    if ship.command_token:
        start_x, start_y = to_template_coord((-ship.base_size[0] / 2, -ship.base_size[1] - 24))
        for i, command in enumerate(ship.command_token):
            token_text = str(command)
            pos_y = start_y + i * 12
            draw.text((start_x, pos_y), token_text, font=font, fill='white', anchor="ls")

    return template_img

def visualize(game : "Armada", title : str,  maneuver_tool : list[tuple[float, float]] | None = None) -> None:
    img = Image.new('RGB', (game.player_edge, game.short_edge), (0,0,0))
    draw = ImageDraw.Draw(img)

    def transform_coord(coord):
        return (coord[0], game.short_edge - coord[1])


    font = ImageFont.truetype("ARIAL.TTF", 18)
    font_small = ImageFont.truetype("ARIAL.TTF", 12)


    draw.text((10, 10), title, font=font, fill='white')

    ship_templates = {}
    for ship in game.ships:
        if ship.ship_id not in ship_templates:
             ship_templates[ship.ship_id] = _draw_ship_template(ship, font_small)

    for ship in game.ships:
        if ship.destroyed:
            continue

        template = ship_templates[ship.ship_id]
        
        # Negate angle for Pillow's counter-clockwise rotation
        angle_deg = -math.degrees(ship.orientation)
        
        # Rotate the template around its center (the pivot point)
        rotated_template = template.rotate(angle_deg, expand=True, resample=Image.Resampling.BICUBIC)

        # The ship's (x,y) is the pivot, so we transform it to image coordinates
        img_pivot_x, img_pivot_y = transform_coord((ship.x, ship.y))
        
        # Calculate the top-left corner for pasting by aligning the rotated template's center
        # with the ship's transformed pivot point.
        paste_x = int(img_pivot_x - rotated_template.width / 2)
        paste_y = int(img_pivot_y - rotated_template.height / 2)

        img.paste(rotated_template, (paste_x, paste_y), rotated_template)

    if maneuver_tool:
        transformed_tool_path = [transform_coord(p) for p in maneuver_tool]
        draw.line(transformed_tool_path, fill='grey', width=int(ship_module.TOOL_WIDTH_HALF * 2))
        
        radius = ship_module.TOOL_WIDTH_HALF
        for i, point in enumerate(maneuver_tool):
            if i % 2 != 0:
                transformed_point = transform_coord(point)
                bounding_box = [
                    (transformed_point[0] - radius, transformed_point[1] - radius),
                    (transformed_point[0] + radius, transformed_point[1] + radius)
                ]
                draw.ellipse(bounding_box, fill='darkgrey')

    output_dir = "game_visuals"
    os.makedirs(output_dir, exist_ok=True)
    img.save(os.path.join(output_dir, f'game_state_{game.image_counter}.png'))
    game.image_counter += 1

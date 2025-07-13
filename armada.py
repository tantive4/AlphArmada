import numpy as np
import model
from ship import *
from squadron import *
from dice import *
from PIL import Image, ImageDraw, ImageFont



class Armada:
    def __init__(self):
        self.player_edge = 900
        self.short_edge = 900
        self.ships = []  # max 3 total, 2 + 1

        self.round = 1
        self.winner = None
        self.image_counter = 0



    def deploy_ship(self, ship, x, y, orientation, speed):
        self.ships.append(ship)
        ship.deploy(self, x, y, orientation, speed, len(self.ships) - 1)

    def get_ship_count(self):
        """Counts only the ships that have not been destroyed."""
        return sum(1 for ship in self.ships if not ship.destroyed)

    def ship_phase(self) :
        player = 1
        activation_count = 0

        while activation_count < self.get_ship_count() :
            ship_activate_value = model.choose_ship_activate()

            for ship in self.ships :
                if ship.player != player or ship.activated or ship.destroyed :
                    ship_activate_value[ship.ship_id] = model.MASK_VALUE

            if np.sum(ship_activate_value == model.MASK_VALUE) == len(ship_activate_value):
                player *= -1
                continue

            ship_activate_policy = model.softmax(ship_activate_value)
            ship_to_activate = self.ships[np.argmax(ship_activate_policy)]
            ship_to_activate.activate()
            activation_count += 1
            player *= -1

    def total_destruction(self, player) :
        player_ship_count = sum(1 for ship in self.ships if ship.player == player and not ship.destroyed)
        return player_ship_count == 0


    def status_phase(self) :
        for ship in self.ships :
            if not ship.destroyed : ship.activated = False
        if self.total_destruction(1) : self.winner = -1
        if self.total_destruction(-1) : self.winner = 1
    
    def get_point(self, player) :
        return sum(ship.point for ship in self.ships if ship.player != player and ship.destroyed)
    
    def play_round(self) :
        self.visualize(f'ROUND {self.round} started')
        self.ship_phase()
        self.status_phase()

    def play(self) :
        while self.round <= 6 :
            self.play_round()

            if self.winner != None :
                break
            self.round += 1

        if self.winner == None :
            self.winner = 1 if self.get_point(1) > self.get_point(-1) else -1

        self.visualize(f'Player {self.winner} has won!')


    def visualize(self, title):
        """Creates and saves an image of the current game state."""
        img = Image.new('RGB', (self.player_edge, self.short_edge), 'black')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        # Add the round number to the title
        display_title = f"Round {self.round} | {title}"
        draw.text((10, 10), display_title, font=font, fill='white')

        for ship in self.ships:
            if ship.destroyed:
                continue

            # Draw the ship's base
            base_coords = [ship.front_left_base, ship.front_right_base, ship.rear_right_base, ship.rear_left_base]
            draw.polygon(base_coords, outline='white')

            # Draw the firing arcs
            draw.line([ship.front_arc_center, ship.front_left_arc], fill='yellow')
            draw.line([ship.front_arc_center, ship.front_right_arc], fill='yellow')
            draw.line([ship.rear_arc_center, ship.rear_left_arc], fill='yellow')
            draw.line([ship.rear_arc_center, ship.rear_right_arc], fill='yellow')

            # Ship Name
            draw.text(ship._get_coordination((0,20)), ship.name, font=font, fill='cyan')
            
            # Hull
            draw.text((ship.x - 5, ship.y - 10), str(ship.hull), font=font, fill='green')
            
            # Shields
            # Front
            draw.text(((ship.front_left_base[0] + ship.front_right_base[0]) / 2, (ship.front_left_base[1] + ship.front_right_base[1]) / 2 - 15), str(ship.shield[0]), font=font, fill='cyan')
            # Right
            draw.text(((ship.front_right_base[0] + ship.rear_right_base[0]) / 2 + 5, (ship.front_right_base[1] + ship.rear_right_base[1]) / 2), str(ship.shield[1]), font=font, fill='cyan')
            # Rear
            draw.text(((ship.rear_left_base[0] + ship.rear_right_base[0]) / 2, (ship.rear_left_base[1] + ship.rear_right_base[1]) / 2 + 5), str(ship.shield[2]), font=font, fill='cyan')
            # Left
            draw.text(((ship.front_left_base[0] + ship.rear_left_base[0]) / 2 - 15, (ship.front_left_base[1] + ship.rear_left_base[1]) / 2), str(ship.shield[3]), font=font, fill='cyan')


        img.save(f'game_visualizer/game_state_{self.image_counter}.png')
        self.image_counter += 1
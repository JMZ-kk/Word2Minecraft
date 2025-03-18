import numpy as np
from PIL import Image
import pygame
import sys
import json
import imageio
import os
import argparse
import openai

from utils import map_to_list, find_most_similar_images,extract_list,extract_dict
from solvers import find_characters
from fixers import pad_rows_to_max_length

from configs import Config

# Initialize pygame
pygame.init()
cfg = Config()

# Define constants
TILE_SIZE = 16
CAMERA_WIDTH = 20
CAMERA_HEIGHT = 16
INFO_PANEL_HEIGHT = 4  # Number of tiles for the info panel


parser = argparse.ArgumentParser(description="Process game inputs")
parser.add_argument('--game_path', type=str, help="A path to JSON file of your game. Derfaults to 'word2world\examples\example_1.json'")
args = parser.parse_args()

if args.game_path:
    if not os.path.exists(args.game_path):
            raise ValueError(f"{args.game_path} does not exist. Please provide an existing path.")
    with open(args.game_path, 'r') as file:
        data = json.load(file)
else:
    game = "example_1"
    #game='data_gen_Your_word2world_4'
    game_dir = os.path.join(f"word2world", "examples")

    with open(f'{game_dir}/{game}.json', 'r') as file:
        data = json.load(file)
    # print(os.getcwd())
    # with open('word2world/examples/example_1.json', 'r') as file:
    #     data = json.load(file)

def pick_building_tiles(tiles_1st_layer,tile_map):
    cfg=Config()
    pick_building_prompt = (f'Given a 2D map \n{tile_map}\n, and a dictionary {tiles_1st_layer} where each key is a tile\'s '
                            f'description and each value is the notation for that tile in the map.'
                            f'identify which tile notations in the map need to be scaled?'
                            f'A tile is considered scaled if it should occupy more than one grid cell,'
                            f' such as a \'house\' or similar object.'
                            f'create a Python list of tiles, formatted as a list of tile codes (e.g., `[\'a\',\'b\']`)')
    print(pick_building_prompt)
    pick_building_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": pick_building_prompt}
    ], temperature=1)
    building_tiles_list = extract_list(pick_building_discriptions['choices'][0]['message']['content'])
    print(building_tiles_list)
    print("\n")
    return building_tiles_list, pick_building_discriptions,pick_building_prompt

def extract_building_size(story,pick_building_discriptions, pick_building_prompt):
    cfg=Config()
    extract_size_prompt = ("Assign each tile a size from the options: small, medium,"
                            " or big, formatted as a dict of tile codes (e.g., {'a':'big','b':'small'}),"
                            " without any additional text or explanation.")
    print(pick_building_prompt)
    story_prompt = f"Write a {cfg.story_paragraphs[0]}-{cfg.story_paragraphs[1]} paragraph story which has characters including the protagonist trying to achieve something and the antagonist wanting to stop the protagonist. The story should describe an environment(s) where the story is set up."

    size_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": story_prompt},
        {"role": "assistant", "content": story},
        {"role": "user", "content": pick_building_prompt},
        {"role": "assistant", "content": pick_building_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": extract_size_prompt}
    ],
                                                              temperature=1)
    size_dict = extract_dict(size_discriptions['choices'][0]['message']['content'])
    print(size_dict)
    print("\n")
    return size_dict, size_discriptions,extract_size_prompt

def modify_string(s, k, new_char):
    s_list = list(s)
    s_list[k] = new_char
    return ''.join(s_list)

round_number = "round_0"
character_descriptions_dict = {}
gen_story = data[round_number]["story"]
grid_str = data[round_number]["world"]
print(grid_str)
grid_str = pad_rows_to_max_length(grid_str)
print('---')
print(grid_str)
grid_world = map_to_list(grid_str)

char_tile_mapping = data[round_number]["tile_mapping"]

walkables = data[round_number]["walkable_tiles"]

print(walkables)
print('---')

important_tiles = data[round_number]["important_tiles"]
print('important')
print(important_tiles)

interactive_object_tiles = data[round_number]["interactive_object_tiles"]
goals = data[round_number]["goals"]

print('object_tiles')
print(interactive_object_tiles)
print('---')


world_1st_layer = data[round_number]["world_1st_layer"]["world"]
world_1st_layer = pad_rows_to_max_length(world_1st_layer)
grid_1st_layer = map_to_list(world_1st_layer)

tile_counts = {}
for row in world_1st_layer:
    for tile in row:
        if tile not in tile_counts:
            tile_counts[tile] = 1
        else:
            tile_counts[tile] += 1

sorted_items = sorted(tile_counts.items(), key=lambda x: x[1], reverse=True)
top_half = sorted_items[:len(tile_counts) // 2]
top_half_keys = {k for k, v in top_half}

check=np.zeros((len(grid_1st_layer), len(grid_1st_layer[0])))
for i in range(len(grid_1st_layer)):
    for j in range(len(grid_1st_layer[0])):
        if f'''{grid_world[i][j]}''' not in walkables:
            for m in [-1,0,1]:
                for n in [-1,0,1]:
                    try:
                        if check[i+m][j+n]==0:
                            check[i+m][j+n]=2
                    except:
                        pass
            check[i][j]=1
        elif f'''{grid_world[i][j]}''' in interactive_object_tiles:
            for m in [-1,0,1]:
                for n in [-1,0,1]:
                    try:
                        if check[i+m][j+n]==0:
                            check[i+m][j+n]=2
                    except:
                        pass
            check[i][j]=3

tiles_1st_layer = data[round_number]["world_1st_layer"]["tiles"]

tileset, _s = find_most_similar_images(char_tile_mapping, cfg.tile_data_dir,interactive_object_tiles)

print(tiles_1st_layer)
story=data[round_number]['story']
non_object_dict={k: v for k, v in tiles_1st_layer.items()
                 if v not in interactive_object_tiles}
building_tiles_list, pick_building_discriptions, pick_building_prompt=pick_building_tiles(non_object_dict,world_1st_layer)
print(building_tiles_list)
size_dict, size_discriptions,extract_size_prompt=extract_building_size(story,pick_building_discriptions, pick_building_prompt)
print(size_dict)

for i in range(len(grid_1st_layer)):
    for j in range(len(grid_1st_layer[0])):
        if grid_1st_layer[i][j] in building_tiles_list:
            check[i][j]=4



def tile_evaluation(check,m,n,tile_counts,grid_1st_layer,tar):
    # 0-2: common tiles
    # 3: object
    # 4: house
    # 5: new house
    if check[m][n] == 3 or check[m][n] == 5:
        return -10000
    if check[m][n] == 4:
        if grid_1st_layer[m][n] == tar:
            return 200
        else:
            return -10000
    return tile_counts[grid_1st_layer[m][n]]

def block_evaluation(check,m,n,tile_counts,grid_1st_layer,tar,block_size):
    sum=0
    for i in range(block_size):
        for j in range(block_size):
            sum+=tile_evaluation(check,m+i,n+j,tile_counts,grid_1st_layer,tar)
    return sum

def update_map(check,biggest_m,biggest_n,block_size,tar):
    for i in range(block_size):
        for j in range(block_size):
            check[biggest_m+i][biggest_n+j] = 5
            grid_1st_layer[biggest_m + i] = modify_string(grid_1st_layer[biggest_m + i], biggest_n+j,
                                                          tar)


for i in range(len(grid_1st_layer)):
    for j in range(len(grid_1st_layer[0])):
        if grid_1st_layer[i][j] in building_tiles_list:
            size=size_dict[grid_1st_layer[i][j]]
            tar=grid_1st_layer[i][j]
            if size=='medium':
                block_size=2
            elif size=='big':
                block_size=3
            else:
                continue
            biggest_m=-1
            biggest_n=-1
            biggest=0
            for m in [i-1,i]:
                for n in [j-1,j]:
                    try:
                        tsum=block_evaluation(check,m,n,tile_counts,grid_1st_layer,tar,block_size)
                        if tsum>biggest:
                            biggest=tsum
                            biggest_n=n
                            biggest_m=m
                    except:
                        pass
            if biggest_m>-1:
                update_map(check, biggest_m, biggest_n, block_size, tar)
print(grid_1st_layer)



WIDTH = CAMERA_WIDTH * TILE_SIZE
HEIGHT = (CAMERA_HEIGHT + INFO_PANEL_HEIGHT) * TILE_SIZE

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Word2World Game")

character_chars = find_characters(grid_str)

# Player setup
player_pos = [character_chars['@'][0], character_chars['@'][1]]  # Starting position of the player
player_size = TILE_SIZE

camera_pos = [max(0, player_pos[0] - CAMERA_WIDTH // 2), max(0, player_pos[1] - CAMERA_HEIGHT // 2)]

picked_objects = {}  # Dictionary to keep track of picked objects

# Enemy setup
enemy_pos = [character_chars['#'][0], character_chars['#'][1]]  # Starting position of the enemy
enemy_direction = 1  # Enemy direction: 1 for right, -1 for left
enemy_bullets = []  # List to store enemy bullets
player_bullets = []  # List to store player bullets

def pil_to_pygame(pil_image):
    """Convert a PIL image to a Pygame surface."""
    mode = pil_image.mode
    size = pil_image.size
    data = pil_image.tobytes()
    return pygame.image.fromstring(data, size, mode).convert_alpha()

tile_counts = {}
for row in world_1st_layer:
    for tile in row:
        if f'''{tile}''' in walkables :
            if tile not in tile_counts:
                tile_counts[tile] = 1
            else:
                tile_counts[tile] += 1
default_walkable_tile = max(tile_counts, key=tile_counts.get)
print(default_walkable_tile)


def draw_map():
    for y in range(CAMERA_HEIGHT):
        for x in range(CAMERA_WIDTH):
            world_x = x + camera_pos[0]
            world_y = y + camera_pos[1]
            if world_y >= len(grid_1st_layer) or world_x >= len(grid_1st_layer[0]):
                continue
            tile = grid_1st_layer[world_y][world_x]
            if tile == '@' or tile == '#':
                pass
            else:
                pil_image = tileset.get(default_walkable_tile)
                if pil_image is not None:
                    pygame_surface = pil_to_pygame(pil_image)
                    screen.blit(pygame_surface, (x * TILE_SIZE, y * TILE_SIZE))
                else:
                    pygame.draw.rect(screen, (255, 0, 255), (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
    for y in range(CAMERA_HEIGHT):
        for x in range(CAMERA_WIDTH):
            world_x = x + camera_pos[0]
            world_y = y + camera_pos[1]
            if world_y >= len(grid_1st_layer) or world_x >= len(grid_1st_layer[0]):
                continue
            tile = grid_1st_layer[world_y][world_x]
            if tile == '@' or tile == '#':
                pass
            else:
                pil_image = tileset.get(tile)
                if pil_image is not None:
                    pygame_surface = pil_to_pygame(pil_image)
                    screen.blit(pygame_surface, (x * TILE_SIZE, y * TILE_SIZE))
                else:
                    pygame.draw.rect(screen, (255, 0, 255), (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
    for y in range(CAMERA_HEIGHT):
        for x in range(CAMERA_WIDTH):
            world_x = x + camera_pos[0]
            world_y = y + camera_pos[1]
            if world_y >= len(grid_world) or world_x >= len(grid_world[0]):
                continue
            tile = grid_world[world_y][world_x]
            if tile == '@' or tile == '#':
                pass
            else:
                pil_image = tileset.get(tile)
                if pil_image is not None:
                    pygame_surface = pil_to_pygame(pil_image)
                    screen.blit(pygame_surface, (x * TILE_SIZE, y * TILE_SIZE))
                else:
                    pygame.draw.rect(screen, (255, 0, 255), (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))

def draw_info_panel():
    y_offset = CAMERA_HEIGHT * TILE_SIZE

    # Clear the info panel area
    pygame.draw.rect(screen, (0, 0, 0), (0, y_offset, WIDTH, INFO_PANEL_HEIGHT * TILE_SIZE))

    for i, (obj, count) in enumerate(picked_objects.items()):
        pil_image = tileset.get(obj)
        if pil_image is not None:
            pygame_surface = pil_to_pygame(pil_image)
            screen.blit(pygame_surface, (i * TILE_SIZE, y_offset))
            font = pygame.font.Font(None, 24)
            text = font.render(str(count), True, (255, 255, 255))
            screen.blit(text, (i * TILE_SIZE, y_offset + TILE_SIZE))

# Load the player image using Pillow
player_image_pil = tileset['@']
player_image_surface = pil_to_pygame(player_image_pil)

# Load the enemy image using Pillow
enemy_image_pil = tileset['#']
enemy_image_surface = pil_to_pygame(enemy_image_pil)

def draw_player():
    screen.blit(player_image_surface, ((player_pos[0] - camera_pos[0]) * TILE_SIZE, (player_pos[1] - camera_pos[1]) * TILE_SIZE))

def draw_enemy():
    screen.blit(enemy_image_surface, ((enemy_pos[0] - camera_pos[0]) * TILE_SIZE, (enemy_pos[1] - camera_pos[1]) * TILE_SIZE))

def draw_bullets():
    for bullet in enemy_bullets:
        pygame.draw.circle(screen, (255, 255, 255), ((bullet[0] - camera_pos[0]) * TILE_SIZE + TILE_SIZE // 2, (bullet[1] - camera_pos[1]) * TILE_SIZE + TILE_SIZE // 2), 3)
    
    
    player_bullet_color = (255, 165, 0) 
    for bullet in player_bullets:
        bullet_x = (bullet[0] - camera_pos[0]) * TILE_SIZE + TILE_SIZE // 2
        bullet_y = (bullet[1] - camera_pos[1]) * TILE_SIZE + TILE_SIZE // 2
        bullet_direction = (bullet[2], bullet[3])

        
        if bullet_direction == (1, 0):  # Right
            points = [(bullet_x, bullet_y), (bullet_x + 6, bullet_y - 3), (bullet_x + 9, bullet_y), (bullet_x + 6, bullet_y + 3)]
        elif bullet_direction == (-1, 0):  # Left
            points = [(bullet_x, bullet_y), (bullet_x - 6, bullet_y - 3), (bullet_x - 9, bullet_y), (bullet_x - 6, bullet_y + 3)]
        elif bullet_direction == (0, 1):  # Down
            points = [(bullet_x, bullet_y), (bullet_x - 3, bullet_y + 6), (bullet_x, bullet_y + 9), (bullet_x + 3, bullet_y + 6)]
        elif bullet_direction == (0, -1):  # Up
            points = [(bullet_x, bullet_y), (bullet_x - 3, bullet_y - 6), (bullet_x, bullet_y - 9), (bullet_x + 3, bullet_y - 6)]

        pygame.draw.polygon(screen, player_bullet_color, points)
def move_player(dx, dy):
    new_x = player_pos[0] + dx
    new_y = player_pos[1] + dy

    # Check if the new position is out of bounds
    if new_x < 0 or new_x >= len(grid_world[0]) or new_y < 0 or new_y >= len(grid_world):
        return False  # Don't move if out of bounds

    # Check for collisions
    if f'''{grid_world[new_y][new_x]}''' not in walkables:
        return False  # Can't move into walls

    player_pos[0] = new_x
    player_pos[1] = new_y
    update_camera()

    # Check for interaction
    pick_object()

    return True

def update_camera():
    camera_pos[0] = max(0, min(player_pos[0] - CAMERA_WIDTH // 2, len(grid_world[0]) - CAMERA_WIDTH))
    camera_pos[1] = max(0, min(player_pos[1] - CAMERA_HEIGHT // 2, len(grid_world) - CAMERA_HEIGHT))

def pick_object():
    x, y = player_pos
    target_tile = grid_world[y][x]

    if target_tile in interactive_object_tiles:
        print("Picked an object!")
        grid_world[y] = grid_world[y][:x] + default_walkable_tile + grid_world[y][x + 1:]  # Replace with floor
        if target_tile in picked_objects:
            picked_objects[target_tile] += 1
        else:
            picked_objects[target_tile] = 1

initial_enemy_x = enemy_pos[0]

def move_enemy():
    global enemy_direction
    new_x = enemy_pos[0] + enemy_direction

    # Check if the new position is beyond 5 tiles to the left or right of the initial position
    if abs(new_x - initial_enemy_x) > 5:
        enemy_direction *= -1  # Change direction if the boundary is reached
        return

    # Check if the new position is out of bounds or collides with a wall
    if new_x < 0 or new_x >= len(grid_world[0]) or f'''{grid_world[enemy_pos[1]][new_x]}''' not in walkables:
        enemy_direction *= -1  # Change direction if it hits a boundary or a non-walkable tile
    else:
        enemy_pos[0] = new_x  # Update the enemy's position if it's a valid move

def enemy_detect_player():
    if abs(player_pos[0] - enemy_pos[0]) <= 3 and player_pos[1] == enemy_pos[1]:
        return True
    if abs(player_pos[1] - enemy_pos[1]) <= 3 and player_pos[0] == enemy_pos[0]:
        return True
    return False

def enemy_attack_player():
    if player_pos[0] < enemy_pos[0]:
        enemy_bullets.append([enemy_pos[0], enemy_pos[1], -1, 0])  # Left
    elif player_pos[0] > enemy_pos[0]:
        enemy_bullets.append([enemy_pos[0], enemy_pos[1], 1, 0])  # Right
    elif player_pos[1] < enemy_pos[1]:
        enemy_bullets.append([enemy_pos[0], enemy_pos[1], 0, -1])  # Up
    elif player_pos[1] > enemy_pos[1]:
        enemy_bullets.append([enemy_pos[0], enemy_pos[1], 0, 1])  # Down

def player_shoot():
    direction_offsets = {
            pygame.K_a: (-1, 0),  # Move left
            pygame.K_d: (1, 0),   # Move right
            pygame.K_w: (0, -1),  # Move up
            pygame.K_s: (0, 1)    # Move down
        }
    keys = pygame.key.get_pressed()
    for key, (dx, dy) in direction_offsets.items():
        if keys[key]:
            player_bullets.append([player_pos[0], player_pos[1], dx, dy])

def move_bullets():
    global running
    for bullet in enemy_bullets[:]:
        bullet[0] += bullet[2]
        bullet[1] += bullet[3]
        if bullet[0] == player_pos[0] and bullet[1] == player_pos[1]:
            print("Player hit!")
            running = False  # End the game if the player is hit

        if bullet[0] < 0 or bullet[0] >= len(grid_world[0]) or bullet[1] < 0 or bullet[1] >= len(grid_world) or f'''{grid_world[bullet[1]][bullet[0]]}''' not in walkables:
            enemy_bullets.remove(bullet)
    for bullet in player_bullets[:]:
        bullet[0] += bullet[2]
        bullet[1] += bullet[3]
        if bullet[0] == enemy_pos[0] and bullet[1] == enemy_pos[1]:
            print("Enemy hit!")
            enemy_pos[0], enemy_pos[1] = -1, -1  
            player_bullets.remove(bullet)
        if bullet[0] < 0 or bullet[0] >= len(grid_world[0]) or bullet[1] < 0 or bullet[1] >= len(grid_world) or f'''{grid_world[bullet[1]][bullet[0]]}''' not in walkables:
            player_bullets.remove(bullet)

def hit_enemy():
    
    direction_offsets = {
            pygame.K_a: (-1, 0),  # Move left
            pygame.K_d: (1, 0),   # Move right
            pygame.K_w: (0, -1),  # Move up
            pygame.K_s: (0, 1)    # Move down
        }
    
    keys = pygame.key.get_pressed()
    for key, (dx, dy) in direction_offsets.items():
        if keys[key]:
            x = player_pos[0] + dx
            y = player_pos[1] + dy
            if 0 <= x < len(grid_world[0]) and 0 <= y < len(grid_world) and (x, y) == (enemy_pos[0], enemy_pos[1]):
                print("Hit an enemy!")
                enemy_pos[0], enemy_pos[1] = -1, -1  
                break


# Game loop
running = True
clock = pygame.time.Clock()
move_direction = None
shooting = False

frames = []

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                move_direction = pygame.K_a
            elif event.key == pygame.K_d:
                move_direction = pygame.K_d
            elif event.key == pygame.K_w:
                move_direction = pygame.K_w
            elif event.key == pygame.K_s:
                move_direction = pygame.K_s
            if event.key == pygame.K_SPACE:
                hit_enemy()
            if event.key == pygame.K_z:
                shooting = True
        if event.type == pygame.KEYUP:
            if event.key == move_direction:
                move_direction = None
            if event.key == pygame.K_z:
                shooting = False

    if move_direction:
        direction_offsets = {
            pygame.K_a: (-1, 0),  # Move left
            pygame.K_d: (1, 0),   # Move right
            pygame.K_w: (0, -1),  # Move up
            pygame.K_s: (0, 1)    # Move down
        }
        dx, dy = direction_offsets[move_direction]
        if not move_player(dx, dy):
            move_direction = None

    if shooting:
        player_shoot()

    if enemy_pos[0] != -1 and enemy_pos[1] != -1: 
        if enemy_detect_player():
            enemy_attack_player()
        else:
            move_enemy()

    move_bullets()

    draw_map()
    draw_player()
    draw_enemy()
    draw_bullets()
    draw_info_panel()  
    pygame.display.flip()
    frame = pygame.surfarray.array3d(pygame.display.get_surface())
    frame = frame.transpose([1, 0, 2])
    frames.append(frame)
    clock.tick(10)  

pygame.quit()
sys.exit()
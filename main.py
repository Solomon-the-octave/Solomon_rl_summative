import math
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import DQN

from urban_env_local import UrbanPlanningEnv


pygame.init()

WIDTH, HEIGHT = 1500, 920
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Urban RL City")
clock = pygame.time.Clock()

title_font = pygame.font.SysFont("arial", 30, bold=True)
heading_font = pygame.font.SysFont("arial", 18, bold=True)
font = pygame.font.SysFont("arial", 14)
small_font = pygame.font.SysFont("arial", 12)
tiny_font = pygame.font.SysFont("arial", 10)

# dark palette
BG = (14, 16, 21)
PANEL = (24, 28, 36)
MAP_BG = (30, 35, 44)
TEXT = (235, 239, 245)
MUTED = (150, 160, 173)
OUTLINE = (58, 66, 79)
SOFT = (63, 71, 84)

ROAD = (72, 78, 90)
ROAD_EDGE = (104, 110, 123)
ROAD_LINE = (235, 191, 72)

BLUE = (88, 145, 255)
GREEN = (89, 196, 128)
RED = (239, 98, 98)
ORANGE = (242, 166, 79)
PURPLE = (174, 124, 255)
TEAL = (81, 195, 190)
YELLOW = (248, 209, 84)
GREY = (165, 173, 184)
DARK = (20, 23, 29)

SHADOW = (5, 7, 10)
RIVER = (71, 128, 217)
RIVER_LIGHT = (123, 172, 245)
GRASS = (56, 103, 69)
SAND = (101, 92, 74)

LEFT_W = 255
RIGHT_W = 290
MAP_X = LEFT_W + 18
MAP_Y = 16
MAP_W = WIDTH - LEFT_W - RIGHT_W - 36
MAP_H = HEIGHT - 32

env = UrbanPlanningEnv()
obs, _ = env.reset()

MODEL_PATHS = ["final_best_model_dqn", "final_best_model_dqn.zip"]
model = None
for path in MODEL_PATHS:
    if Path(path).exists():
        model = DQN.load(path, device="cpu")
        break

if model is None:
    raise FileNotFoundError("Could not find final_best_model_dqn or final_best_model_dqn.zip in this folder.")

SITES = {
    0: {"name": "Idle",       "pos": (430, 320), "color": GREY,   "code": "ID"},
    1: {"name": "Housing",    "pos": (250, 215), "color": ORANGE, "code": "H"},
    2: {"name": "Transport",  "pos": (655, 230), "color": RED,    "code": "T"},
    3: {"name": "Water",      "pos": (770, 515), "color": BLUE,   "code": "W"},
    4: {"name": "Settlement", "pos": (235, 570), "color": PURPLE, "code": "U"},
    5: {"name": "Green",      "pos": (525, 675), "color": GREEN,  "code": "G"},
    6: {"name": "Services",   "pos": (640, 425), "color": TEAL,   "code": "S"},
}

ROADS = [
    (0, 1), (0, 2), (0, 4), (0, 6),
    (1, 4),
    (2, 6), (2, 3),
    (3, 6), (3, 5),
    (4, 5),
    (5, 6),
]

ACTION_EFFECT_TEXT = {
    0: "No intervention. Pressure may rise over time.",
    1: "Housing increased. Informal pressure fell. Budget decreased.",
    2: "Transport increased. Pollution fell. Budget decreased.",
    3: "Water access increased. Informal pressure fell. Budget decreased.",
    4: "Settlement upgrade improved housing and water. Budget decreased.",
    5: "Green space increased. Pollution fell. Budget decreased.",
    6: "Services improved. Population pressure fell. Budget decreased.",
}

reward_popups = []
site_pulses = []
path_glow_dots = []

agent_x, agent_y = SITES[0]["pos"]
move_start = SITES[0]["pos"]
move_target = SITES[0]["pos"]
move_progress = 1.0

current_action = 0
work_timer = 0
work_duration_frames = 180
pause_timer = 0
decision_timer = 18
phase = "decide"

last_reward = 0.0
episode_done = False
last_explanation = "Waiting"
prev_metrics = env.get_metrics().copy()


def map_point(x, y):
    return MAP_X + 28 + x, MAP_Y + 68 + y


def draw_text(text, x, y, font_obj, color=TEXT):
    surf = font_obj.render(text, True, color)
    screen.blit(surf, (x, y))


def add_popup(text, x, y, color):
    reward_popups.append({
        "text": text,
        "x": x,
        "y": y,
        "life": 70,
        "color": color
    })


def add_pulse(x, y, color):
    site_pulses.append({
        "x": x,
        "y": y,
        "life": 32,
        "color": color
    })


def rebuild_path_glow():
    path_glow_dots.clear()
    sx, sy = move_start
    tx, ty = move_target
    steps = 10
    for i in range(1, steps):
        t = i / steps
        x = sx + (tx - sx) * t
        y = sy + (ty - sy) * t
        path_glow_dots.append([x, y, i * 6])


def draw_round_panel(x, y, w, h):
    pygame.draw.rect(screen, PANEL, (x, y, w, h), border_radius=18)
    pygame.draw.rect(screen, OUTLINE, (x, y, w, h), 1, border_radius=18)


def draw_bar(x, y, w, h, value, color, label, invert=False):
    value = max(0, min(100, float(value)))
    shown = 100 - value if invert else value
    pygame.draw.rect(screen, SOFT, (x, y, w, h), border_radius=7)
    fill = int((shown / 100.0) * w)
    pygame.draw.rect(screen, color, (x, y, fill, h), border_radius=7)
    pygame.draw.rect(screen, OUTLINE, (x, y, w, h), 1, border_radius=7)
    draw_text(f"{label}: {int(value)}", x, y - 16, small_font, MUTED)


def draw_iso_building(cx, cy, base_w, base_h, height, top_color, side_left, side_right):
    roof_y = cy - base_h // 2 - height

    top = [
        (cx, roof_y),
        (cx + base_w // 2, roof_y + base_h // 2),
        (cx, roof_y + base_h),
        (cx - base_w // 2, roof_y + base_h // 2),
    ]
    left = [
        (cx - base_w // 2, cy),
        (cx, cy + base_h // 2),
        (cx, roof_y + base_h),
        (cx - base_w // 2, roof_y + base_h // 2),
    ]
    right = [
        (cx + base_w // 2, cy),
        (cx, cy + base_h // 2),
        (cx, roof_y + base_h),
        (cx + base_w // 2, roof_y + base_h // 2),
    ]

    shadow_poly = [
        (cx - base_w // 2 + 10, cy + 12),
        (cx + base_w // 2 + 10, cy + 12),
        (cx + 12, cy + base_h // 2 + 12),
        (cx - base_w // 2 + 2, cy + base_h // 2 + 12),
    ]
    pygame.draw.polygon(screen, SHADOW, shadow_poly)

    pygame.draw.polygon(screen, side_left, left)
    pygame.draw.polygon(screen, side_right, right)
    pygame.draw.polygon(screen, top_color, top)

    for poly in (top, left, right):
        pygame.draw.polygon(screen, OUTLINE, poly, 1)


def draw_background():
    pygame.draw.ellipse(screen, GRASS, (MAP_X + 55, MAP_Y + 122, 150, 92))
    pygame.draw.ellipse(screen, GRASS, (MAP_X + 505, MAP_Y + 700, 190, 95))
    pygame.draw.ellipse(screen, GRASS, (MAP_X + 790, MAP_Y + 705, 128, 84))
    pygame.draw.ellipse(screen, SAND, (MAP_X + 800, MAP_Y + 160, 118, 570))

    river = [
        (MAP_X + 842, MAP_Y + 108),
        (MAP_X + 878, MAP_Y + 246),
        (MAP_X + 865, MAP_Y + 424),
        (MAP_X + 829, MAP_Y + 654),
        (MAP_X + 846, MAP_Y + 850),
    ]
    pygame.draw.lines(screen, RIVER, False, river, 30)
    pygame.draw.lines(screen, RIVER_LIGHT, False, river, 10)

    decor = [
        (MAP_X + 135, MAP_Y + 388, 58, 32, 40),
        (MAP_X + 640, MAP_Y + 150, 64, 36, 45),
        (MAP_X + 705, MAP_Y + 624, 66, 36, 50),
        (MAP_X + 395, MAP_Y + 760, 60, 34, 42),
    ]
    for x, y, w, h, ht in decor:
        draw_iso_building(
            x, y, w, h, ht,
            (116, 123, 135),
            (76, 82, 93),
            (67, 73, 83),
        )


def draw_road(a_idx, b_idx):
    ax, ay = map_point(*SITES[a_idx]["pos"])
    bx, by = map_point(*SITES[b_idx]["pos"])

    pygame.draw.line(screen, ROAD_EDGE, (ax, ay), (bx, by), 28)
    pygame.draw.line(screen, ROAD, (ax, ay), (bx, by), 20)

    dx = bx - ax
    dy = by - ay
    dist = math.hypot(dx, dy)
    if dist == 0:
        return

    ux = dx / dist
    uy = dy / dist
    cur = 10

    while cur < dist - 10:
        x1 = ax + ux * cur
        y1 = ay + uy * cur
        x2 = ax + ux * min(cur + 12, dist)
        y2 = ay + uy * min(cur + 12, dist)
        pygame.draw.line(screen, ROAD_LINE, (x1, y1), (x2, y2), 2)
        cur += 26


def draw_roads():
    for a, b in ROADS:
        draw_road(a, b)


def draw_path_glow():
    if phase not in ("move", "work"):
        return

    for dot in path_glow_dots:
        dot[2] -= 1
        if dot[2] <= 0:
            dot[2] = 60

        alpha_factor = dot[2] / 60.0
        x, y = map_point(dot[0], dot[1])
        radius = 4 + int(2 * alpha_factor)
        pygame.draw.circle(screen, (255, 220, 110), (int(x), int(y)), radius)


def metric_to_height(value, min_h=22, max_h=74):
    return int(min_h + (value / 100.0) * (max_h - min_h))


def draw_sites():
    metrics = env.get_metrics()

    site_values = {
        1: metrics["Housing"],
        2: metrics["Transport"],
        3: metrics["Water and Sanitation"],
        4: 100 - metrics["Informal Settlements"],
        5: metrics["Green Space"],
        6: 100 - metrics["Pollution"],
        0: metrics["Budget"],
    }

    for idx, site in SITES.items():
        x, y = map_point(*site["pos"])
        value = site_values[idx]
        h = metric_to_height(value)

        top = (205, 210, 218)
        left = (87, 94, 106)
        right = (74, 81, 92)

        if idx == 1:
            top = (197, 151, 98)
            left = (153, 113, 71)
            right = (135, 99, 60)
        elif idx == 2:
            top = (193, 108, 108)
            left = (152, 81, 81)
            right = (132, 69, 69)
        elif idx == 3:
            top = (113, 149, 216)
            left = (80, 115, 177)
            right = (67, 102, 159)
        elif idx == 4:
            top = (161, 119, 215)
            left = (125, 88, 171)
            right = (110, 77, 153)
        elif idx == 5:
            top = (110, 183, 120)
            left = (80, 144, 88)
            right = (69, 127, 77)
        elif idx == 6:
            top = (92, 177, 170)
            left = (67, 138, 132)
            right = (56, 122, 116)
        elif idx == 0:
            top = (166, 174, 186)
            left = (114, 122, 134)
            right = (98, 107, 118)

        draw_iso_building(x, y, 80, 42, h, top, left, right)

        ring_radius = 44 + int((value / 100.0) * 10)
        pulse_boost = 0
        if idx == current_action:
            pulse_boost = int(abs(math.sin(pygame.time.get_ticks() * 0.004)) * 8)

        pygame.draw.circle(screen, site["color"], (int(x), int(y - h - 4)), ring_radius + pulse_boost, 2)
        draw_text(site["code"], x - 10, y - h - 18, heading_font, site["color"])

        if idx != 0:
            draw_text(site["name"], x - 24, y + 28, tiny_font, MUTED)


def draw_agent():
    x, y = map_point(agent_x, agent_y)

    pygame.draw.ellipse(screen, SHADOW, (x - 20, y + 13, 40, 12))
    pygame.draw.rect(screen, BLUE, (x - 18, y - 10, 36, 18), border_radius=6)
    pygame.draw.rect(screen, (61, 103, 179), (x - 6, y - 18, 18, 12), border_radius=4)
    pygame.draw.rect(screen, (220, 231, 244), (x - 1, y - 14, 8, 6), border_radius=2)
    pygame.draw.circle(screen, DARK, (int(x - 10), int(y + 8)), 4)
    pygame.draw.circle(screen, DARK, (int(x + 10), int(y + 8)), 4)
    pygame.draw.circle(screen, YELLOW, (int(x + 18), int(y - 1)), 3)

    glow_r = 22 + int(abs(math.sin(pygame.time.get_ticks() * 0.005)) * 4)
    pygame.draw.circle(screen, (110, 160, 255), (int(x), int(y)), glow_r, 2)

    if phase == "work":
        tool_angle = math.sin(pygame.time.get_ticks() * 0.02) * 0.9
        tx = x + 24
        ty = y - 22
        tx2 = tx + math.cos(tool_angle) * 11
        ty2 = ty + math.sin(tool_angle) * 11
        pygame.draw.line(screen, YELLOW, (tx, ty), (tx2, ty2), 3)
        pygame.draw.circle(screen, YELLOW, (int(tx2), int(ty2)), 4)


def draw_popups():
    for popup in reward_popups[:]:
        popup["y"] -= 0.8
        popup["life"] -= 1
        draw_text(popup["text"], popup["x"], popup["y"], small_font, popup["color"])
        if popup["life"] <= 0:
            reward_popups.remove(popup)


def draw_pulses():
    for pulse in site_pulses[:]:
        pulse["life"] -= 1
        radius = 22 + (32 - pulse["life"]) * 3
        pygame.draw.circle(screen, pulse["color"], (int(pulse["x"]), int(pulse["y"])), radius, 3)
        if pulse["life"] <= 0:
            site_pulses.remove(pulse)


def draw_map():
    draw_round_panel(MAP_X, MAP_Y, MAP_W, MAP_H)
    pygame.draw.rect(screen, MAP_BG, (MAP_X + 12, MAP_Y + 12, MAP_W - 24, MAP_H - 24), border_radius=18)
    pygame.draw.rect(screen, OUTLINE, (MAP_X + 12, MAP_Y + 12, MAP_W - 24, MAP_H - 24), 1, border_radius=18)

    draw_text("Urban planning simulation", MAP_X + 28, MAP_Y + 24, heading_font)

    draw_background()
    draw_roads()
    draw_path_glow()
    draw_pulses()
    draw_sites()
    draw_agent()
    draw_popups()


def draw_left_panel():
    draw_round_panel(18, 18, LEFT_W, HEIGHT - 36)
    metrics = env.get_metrics()

    draw_text("Urban RL City", 34, 34, title_font)

    y = 98
    draw_bar(34, y, 200, 12, metrics["Housing"], ORANGE, "Housing")
    y += 40
    draw_bar(34, y, 200, 12, metrics["Transport"], RED, "Transport")
    y += 40
    draw_bar(34, y, 200, 12, metrics["Water and Sanitation"], BLUE, "Water")
    y += 40
    draw_bar(34, y, 200, 12, metrics["Green Space"], GREEN, "Green")
    y += 40
    draw_bar(34, y, 200, 12, metrics["Informal Settlements"], PURPLE, "Informal", invert=True)
    y += 40
    draw_bar(34, y, 200, 12, metrics["Pollution"], TEAL, "Pollution", invert=True)
    y += 40
    draw_bar(34, y, 200, 12, metrics["Budget"], (136, 143, 151), "Budget")

    draw_text("Run Info", 34, 424, heading_font)
    draw_text(f"Step: {env.current_step}/{env.max_steps}", 34, 456, font, MUTED)
    draw_text(f"Action: {env.action_names[current_action]}", 34, 484, font, MUTED)
    draw_text(f"Reward: {last_reward:.2f}", 34, 512, font, GREEN if last_reward >= 0 else RED)
    draw_text("Model: trained DQN", 34, 540, small_font, BLUE)

    status = "Waiting"
    status_color = GREEN
    if phase == "move":
        status = "Moving to site"
        status_color = BLUE
    elif phase == "work":
        status = "Applying action"
        status_color = ORANGE
    elif phase == "pause":
        status = "Updating state"
        status_color = PURPLE

    draw_text(f"Status: {status}", 34, 568, font, status_color)

    draw_text("Reward change", 34, 618, heading_font)
    explanation_lines = [
        last_explanation[:34],
        last_explanation[34:68],
        last_explanation[68:102]
    ]
    yy = 650
    for line in explanation_lines:
        if line.strip():
            draw_text(line.strip(), 34, yy, small_font, MUTED)
            yy += 19

    progress = 0
    if phase == "work":
        progress = (work_timer / work_duration_frames) * 100
    draw_bar(34, 742, 200, 12, progress, BLUE, "Progress")

    draw_text("Keys", 34, 800, heading_font)
    draw_text("R - reset", 34, 832, small_font, MUTED)
    draw_text("ESC - quit", 34, 852, small_font, MUTED)


def draw_right_panel():
    draw_round_panel(WIDTH - RIGHT_W - 18, 18, RIGHT_W, HEIGHT - 36)
    x = WIDTH - RIGHT_W
    metrics = env.get_metrics()

    draw_text("Environment State", x, 34, heading_font)

    cards = [
        ("Population Pressure", metrics["Population Pressure"], RED),
        ("Housing", metrics["Housing"], ORANGE),
        ("Transport", metrics["Transport"], RED),
        ("Water and Sanitation", metrics["Water and Sanitation"], BLUE),
        ("Green Space", metrics["Green Space"], GREEN),
        ("Informal Settlements", metrics["Informal Settlements"], PURPLE),
        ("Pollution", metrics["Pollution"], TEAL),
        ("Budget", metrics["Budget"], (136, 143, 151)),
    ]

    y = 78
    for label, value, color in cards:
        pygame.draw.rect(screen, (30, 34, 41), (x, y, 225, 56), border_radius=12)
        pygame.draw.rect(screen, OUTLINE, (x, y, 225, 56), 1, border_radius=12)
        draw_text(label, x + 14, y + 10, small_font, MUTED)
        pygame.draw.rect(screen, SOFT, (x + 14, y + 31, 160, 8), border_radius=4)
        pygame.draw.rect(screen, color, (x + 14, y + 31, int((value / 100.0) * 160), 8), border_radius=4)
        draw_text(str(int(value)), x + 182, y + 22, small_font, TEXT)
        y += 64

    draw_text("How it works", x, 638, heading_font)
    draw_text("1. DQN picks an action", x + 8, 672, small_font, MUTED)
    draw_text("2. Agent moves to that site", x + 8, 692, small_font, MUTED)
    draw_text("3. env.step(action) runs", x + 8, 712, small_font, MUTED)
    draw_text("4. State and reward update", x + 8, 732, small_font, MUTED)


def reset_episode():
    global obs, agent_x, agent_y, move_start, move_target, move_progress
    global current_action, work_timer, pause_timer, decision_timer
    global phase, last_reward, episode_done, prev_metrics, last_explanation

    obs, _ = env.reset()
    prev_metrics = env.get_metrics().copy()

    agent_x, agent_y = SITES[0]["pos"]
    move_start = SITES[0]["pos"]
    move_target = SITES[0]["pos"]
    move_progress = 1.0

    current_action = 0
    work_timer = 0
    pause_timer = 0
    decision_timer = 18
    phase = "decide"
    last_reward = 0.0
    episode_done = False
    last_explanation = "Waiting"

    reward_popups.clear()
    site_pulses.clear()
    path_glow_dots.clear()


def choose_action_from_model():
    global current_action, move_start, move_target, move_progress, phase, last_explanation

    action, _ = model.predict(np.array(obs, dtype=np.float32), deterministic=True)
    action = int(action)

    current_action = action
    move_start = (agent_x, agent_y)
    move_target = SITES[action]["pos"]
    move_progress = 0.0
    phase = "move"
    last_explanation = ACTION_EFFECT_TEXT[action]
    rebuild_path_glow()


def update_move():
    global agent_x, agent_y, move_progress, phase, work_timer

    move_progress += 0.07
    if move_progress >= 1.0:
        move_progress = 1.0
        agent_x, agent_y = move_target
        phase = "work"
        work_timer = 0
        return

    sx, sy = move_start
    tx, ty = move_target
    agent_x = sx + (tx - sx) * move_progress
    agent_y = sy + (ty - sy) * move_progress


def apply_env_action():
    global obs, last_reward, phase, pause_timer, episode_done, prev_metrics, last_explanation

    before = prev_metrics.copy()
    new_obs, reward, terminated, truncated, info = env.step(current_action)
    obs = new_obs.copy()

    after = env.get_metrics().copy()
    last_reward = reward
    episode_done = terminated or truncated

    px, py = map_point(*move_target)

    if reward >= 0:
        add_popup(f"+{reward:.1f}", px + 18, py - 24, GREEN)
        add_pulse(px, py, GREEN)
    else:
        add_popup(f"{reward:.1f}", px + 18, py - 24, RED)
        add_pulse(px, py, RED)

    diffs = []
    tracked = [
        "Housing",
        "Transport",
        "Water and Sanitation",
        "Green Space",
        "Informal Settlements",
        "Pollution",
        "Budget"
    ]

    for key in tracked:
        diff = int(after[key] - before[key])
        if diff != 0:
            sign = "+" if diff > 0 else ""
            diffs.append(f"{key} {sign}{diff}")

    if diffs:
        last_explanation = " | ".join(diffs[:3])

    prev_metrics = after.copy()
    pause_timer = 40
    phase = "pause"


reset_episode()

running = True

while running:
    clock.tick(60)
    screen.fill(BG)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                reset_episode()

    if phase == "decide":
        if episode_done:
            reset_episode()
        else:
            decision_timer -= 1
            if decision_timer <= 0:
                decision_timer = 18
                choose_action_from_model()

    elif phase == "move":
        update_move()

    elif phase == "work":
        work_timer += 1
        if work_timer >= work_duration_frames:
            apply_env_action()

    elif phase == "pause":
        pause_timer -= 1
        if pause_timer <= 0:
            if episode_done:
                reset_episode()
            else:
                phase = "decide"

    draw_left_panel()
    draw_map()
    draw_right_panel()
    pygame.display.flip()

pygame.quit()
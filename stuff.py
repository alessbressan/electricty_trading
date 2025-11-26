import time
import random
import math
import json

import codey
import rocky

try:
    import urequests
except ImportError:
    import requests as urequests


# ============================================================
# ACTIONS / KINEMATICS
# ============================================================

ACTION_FORWARD = 0
ACTION_BACKUP_TURN_LEFT = 1
ACTION_BACKUP_TURN_RIGHT = 2
ACTION_STOP = 3
ACTION_SIZE = 4

ACTION_NAMES = ["FWD", "BCK_L", "BCK_R", "STOP"]

FORWARD_SPEED = 40
BACKUP_SPEED = 50
FORWARD_TIME = 0.10
BACKUP_TIME = 0.20
TURN_TIME = 0.25


# ============================================================
# COLOR CALIBRATION (REFERENCE-STYLE)
# ============================================================

COLOR_TOL_SQ = 900
color_profiles = {"CTR": None, "LFT": None, "RGT": None, "WIN": None}


def smart_display(msg):
    try:
        codey.display.show(str(msg))
    except Exception:
        pass


def _sample_rgb(samples=12, delay=0.04):
    s_r = s_g = s_b = 0
    for _ in range(samples):
        try:
            r = rocky.color_ir_sensor.get_red()
            g = rocky.color_ir_sensor.get_green()
            b = rocky.color_ir_sensor.get_blue()
        except Exception:
            r, g, b = 0, 0, 0
        s_r += r
        s_g += g
        s_b += b
        time.sleep(delay)
    return (s_r // samples, s_g // samples, s_b // samples)


def press_any_key():
    while not (
        codey.button_a.is_pressed()
        or codey.button_b.is_pressed()
        or codey.button_c.is_pressed()
    ):
        time.sleep(0.05)
    time.sleep(0.25)


def calibrate_color(label, samples=12):
    smart_display(label[0].upper())
    press_any_key()
    time.sleep(0.1)
    rgb = _sample_rgb(samples)
    color_profiles[label] = rgb
    smart_display("RDY")
    time.sleep(0.25)


def calibrate_colors():
    time.sleep(0.15)
    for label in ("CTR", "LFT", "RGT", "WIN"):
        calibrate_color(label)
    time.sleep(0.3)
    smart_display("DONE")
    time.sleep(0.5)


def _dist_sq(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _match_color_rgb(r, g, b):
    if any(v is None for v in color_profiles.values()):
        return None
    cur = (r, g, b)
    best = None
    best_d = None
    for label, tpl in color_profiles.items():
        d = _dist_sq(cur, tpl)
        if best is None or d < best_d:
            best = label
            best_d = d
    if best_d is not None and best_d <= COLOR_TOL_SQ:
        return best
    return None


def get_zone():
    # black = goal
    try:
        if rocky.color_ir_sensor.get_gray() < 40:
            return "WIN"
    except Exception:
        pass

    # calibrated RGB
    try:
        r = rocky.color_ir_sensor.get_red()
        g = rocky.color_ir_sensor.get_green()
        b = rocky.color_ir_sensor.get_blue()
        m = _match_color_rgb(r, g, b)
        if m is not None:
            return m
    except Exception:
        pass

    # named colors fallback
    try:
        cmap = {"black": "WIN", "green": "LFT", "blue": "RGT", "white": "CTR"}
        for c, z in cmap.items():
            if rocky.color_ir_sensor.is_color(c):
                return z
    except Exception:
        pass

    return "CTR"


# zone ↔ discrete state index
ZONE_TO_STATE = {"CTR": 0, "LFT": 1, "RGT": 2}
STATE_TO_ZONE = {0: "CTR", 1: "LFT", 2: "RGT"}


def zone_to_state_id(zone):
    return ZONE_TO_STATE.get(zone, 0)


# ============================================================
# WIFI / SERVER UPLOAD (REFERENCE-STYLE)
# ============================================================

WIFI_SSID = "TheSpot"
WIFI_PASS = "@ka1r0u2"
LAPTOP_IP = "192.168.2.121"
SERVER_URL = "http://" + LAPTOP_IP + ":8081/upload"


def connect_wifi():
    smart_display("WUP")
    codey.wifi.start(WIFI_SSID, WIFI_PASS)
    while not codey.wifi.is_connected():
        smart_display("WRK")
        time.sleep(0.2)
    smart_display("WOK")


# ============================================================
# MOVEMENT PRIMITIVES
# ============================================================

def do_forward():
    rocky.forward(FORWARD_SPEED)
    time.sleep(FORWARD_TIME)
    rocky.stop()


def do_backup_turn_left():
    rocky.backward(BACKUP_SPEED)
    time.sleep(BACKUP_TIME)
    rocky.stop()
    rocky.turn_left(BACKUP_SPEED)
    time.sleep(TURN_TIME)
    rocky.stop()


def do_backup_turn_right():
    rocky.backward(BACKUP_SPEED)
    time.sleep(BACKUP_TIME)
    rocky.stop()
    rocky.turn_right(BACKUP_SPEED)
    time.sleep(TURN_TIME)
    rocky.stop()


def do_stop():
    rocky.stop()


ACTION_MAP = {
    ACTION_FORWARD: do_forward,
    ACTION_BACKUP_TURN_LEFT: do_backup_turn_left,
    ACTION_BACKUP_TURN_RIGHT: do_backup_turn_right,
    ACTION_STOP: do_stop,
}


def take_action(action):
    fn = ACTION_MAP.get(action, do_stop)
    fn()


# ============================================================
# REWARD FUNCTION
# ============================================================

def reward_fn(prev_zone, next_zone, action):
    if next_zone == "WIN":
        return 200.0, True

    if prev_zone == "CTR":
        if next_zone == "CTR":
            if action == ACTION_FORWARD:
                return 50.0, False
            else:
                return -200.0, False
        elif next_zone in ("LFT", "RGT"):
            return 0.0, False

    if prev_zone == "LFT":
        if next_zone == "CTR":
            if action == ACTION_BACKUP_TURN_RIGHT:
                return 100.0, False
            elif action == ACTION_BACKUP_TURN_LEFT:
                return -150.0, False
            elif action == ACTION_FORWARD:
                return 20.0, False
            else:
                return -200.0, False
        elif next_zone == "LFT":
            if action == ACTION_FORWARD:
                return -200.0, False
            else:
                return -100.0, False

    if prev_zone == "RGT":
        if next_zone == "CTR":
            if action == ACTION_BACKUP_TURN_LEFT:
                return 100.0, False
            elif action == ACTION_BACKUP_TURN_RIGHT:
                return -150.0, False
            elif action == ACTION_FORWARD:
                return 20.0, False
            else:
                return -200.0, False
        elif next_zone == "RGT":
            if action == ACTION_FORWARD:
                return -200.0, False
            else:
                return -100.0, False

    return 0.0, False


# ============================================================
# ENV WRAPPER (DISCRETE STATE 0/1/2)
# ============================================================

class CodeyEnv:
    def __init__(self, max_steps=120, action_pause=0.05):
        self.max_steps = max_steps
        self.action_pause = action_pause
        self.steps = 0
        self.zone = "CTR"

    def reset(self):
        self.steps = 0
        self.zone = get_zone()
        return zone_to_state_id(self.zone)

    def step(self, action):
        prev_zone = self.zone

        take_action(action)
        time.sleep(self.action_pause)

        self.zone = get_zone()
        self.steps += 1

        reward, terminal = reward_fn(prev_zone, self.zone, action)
        done = terminal or (self.steps >= self.max_steps)

        next_state = zone_to_state_id(self.zone)
        info = {"steps": self.steps, "zone": self.zone}
        return next_state, reward, done, info

    def render(self, episode, step, epsilon):
        print(
            "Ep {} | Step {} | Zone={} | eps={:.2f}".format(
                episode, step, self.zone, epsilon
            )
        )
        smart_display(self.zone)


# ============================================================
# HAND-ROLLED DQN (YOUR FRAMEWORK)
# ============================================================

class DQN:
    def __init__(self, state_dim, action_dim, hidden_dims, lr=0.05, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.lr = lr
        self.gamma = gamma

        self.weights = []
        previous_dim = state_dim

        for hidden_dim in hidden_dims:
            self.weights.append(
                [
                    [random.uniform(-1, 1) for _ in range(hidden_dim)]
                    for _ in range(previous_dim)
                ]
            )
            previous_dim = hidden_dim

        self.weights.append(
            [
                [random.uniform(-1, 1) for _ in range(action_dim)]
                for _ in range(previous_dim)
            ]
        )

    def forward(self, state):
        self.a = []
        self.z = []

        if isinstance(state, int):
            state_list = [0] * self.state_dim
            state_list[state] = 1
            state = state_list

        if not isinstance(state, list):
            state = [state]

        self.a.append(state)

        for i in range(len(self.weights)):
            self.z.append(self.dot(self.a[-1], self.weights[i]))
            self.a.append([self.relu(x) for x in self.z[-1]])

        return self.a[-1][0 : self.action_dim]

    def backward(self, state, action, target):
        if not isinstance(state, list):
            state = [state]

        self.a = [state] + self.a
        deltas = [[0] * len(layer) for layer in self.weights]

        deltas[-1] = [0] * len(self.weights[-1])
        deltas[-1][action] = target - self.a[-1][action]

        for i in reversed(range(len(self.weights) - 1)):
            for k in range(len(self.weights[i + 1])):
                for j in range(len(self.weights[i + 1][k])):
                    if j < len(self.a[i + 1]):
                        deltas[i][j] += (
                            deltas[i + 1][k]
                            * self.weights[i + 1][k][j]
                            * (self.a[i + 1][j] > 0)
                        )

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    if j < len(self.a[i]) and k < len(deltas[i]):
                        self.weights[i][j][k] += (
                            self.lr * self.a[i][j] * deltas[i][k]
                        )

    def update(self, state, action, reward, next_state):
        forward_state = self.forward(state)
        future_rewards = max(self.forward(next_state))
        target = reward + self.lr * (
            reward + self.gamma * future_rewards - forward_state[action]
        )
        self.backward(state, action, target)

    def get_action(self, state, epsilon):
        q_values = self.forward(state)
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return q_values.index(max(q_values))

    def dot(self, a, b):
        if not any(isinstance(i, list) for i in a) and not any(
            isinstance(i, list) for i in b
        ):
            return sum(x * y for x, y in zip(a, b))

        elif all(isinstance(i, list) for i in a) and all(
            isinstance(i, list) for i in b
        ):
            b_t = list(map(list, zip(*b)))
            return [
                [sum(x * y for x, y in zip(row_a, row_b)) for row_b in b_t]
                for row_a in a
            ]

        elif all(isinstance(i, list) for i in b):
            return [
                sum(x * y for x, y in zip(a, neuron_weights))
                for neuron_weights in b
            ]

        else:
            return sum(x * y for x, y in zip(a, b))

    def relu(self, x):
        return max(0, x)

    def outer(self, a, b):
        if isinstance(b, float):
            return [[x * b for _ in range(len(a))] for x in a]
        else:
            return [[x * y for y in b] for x in a]

    def sigmoid(self, x):
        return [1 / (1 + math.exp(-xi)) for xi in x]


# ============================================================
# BUILD & UPLOAD Q-"TABLE" FROM DQN
# ============================================================

def build_q_table_from_dqn(dqn):
    q_table = []
    for state_id in (0, 1, 2):  # CTR, LFT, RGT
        q_vals = dqn.forward(state_id)
        q_table.append([float(v) for v in q_vals])
    return q_table


def upload_q_table(dqn):
    do_stop()
    smart_display("QUP")
    try:
        connect_wifi()
        headers = {"Content-Type": "application/json"}
        q_table = build_q_table_from_dqn(dqn)
        data = json.dumps(q_table)
        res = urequests.post(SERVER_URL, headers=headers, data=data)
        res.close()
        smart_display("QOK")
        time.sleep(0.5)
        smart_display("STP")
    except Exception as e:
        print("[ERROR] Upload failed:", str(e))
        smart_display("Q-F")
        time.sleep(0.5)


# ============================================================
# TRAINING LOOP USING DQN
# ============================================================

def train_on_codey(env, dqn, episodes=5):
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.01  # subtract per episode

    for e in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        step_count = 0
        done = False

        while not done:
            env.render(e, step_count, epsilon)
            action = dqn.get_action(state, epsilon)
            print("Chosen action:", ACTION_NAMES[action])

            next_state, reward, done, info = env.step(action)

            dqn.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward
            step_count += 1

            print("Reward: {:.1f}, Done: {}".format(reward, done))
            time.sleep(0.1)

        if epsilon > epsilon_min:
            epsilon = max(epsilon - epsilon_decay, epsilon_min)

        env.render(e, step_count, epsilon)
        print(
            "Episode {} finished in {} steps | total reward {:.1f} | "
            "epsilon={:.3f}".format(e, step_count, total_reward, epsilon)
        )
        time.sleep(1.0)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    calibrate_colors()
    env = CodeyEnv(max_steps=120, action_pause=0.05)
    dqn = DQN(state_dim=3, action_dim=ACTION_SIZE, hidden_dims=[10, 10],
              lr=0.5, gamma=0.95)

    train_on_codey(env, dqn, episodes=5)

    upload_q_table(dqn)

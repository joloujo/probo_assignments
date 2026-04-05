from enum import Enum
import numpy as np

StateT = tuple[int, int]
class ActionT(Enum):
    up = 0
    down = 1
    left = 2
    right = 3
    stay = 4

GOAL: StateT = (3, 2)
WIDTH: int = 5
HEIGHT: int = 5
DISCOUNT: float = 0.9
CUTOFF: float = 1
REWARD: float = 10
COST: float = -1

state_space = [(x, y) for x in range(WIDTH) for y in range(HEIGHT)]


def add_states(a: StateT, b: StateT) -> StateT:
    c: StateT = tuple(an + bn for (an, bn) in zip(a, b)) # type: ignore
    return c


def transition(state: StateT, action: ActionT, next_state: StateT) -> float:
    match action:
        case ActionT.stay:
            if next_state == state:
                return 0.8
            elif next_state == add_states(state, (0, 1)) or next_state == add_states(state, (0, -1)) or next_state == add_states(state, (1, 0)) or next_state == add_states(state, (-1, 0)):
                return 0.05
            else:
                return 0.0
        case ActionT.up:
            if next_state == add_states(state, (0, -1)):
                return 0.5
            elif next_state == add_states(state, (0, 1)):
                return 0.5
            else:
                return 0.0
        case ActionT.down:
            if next_state == add_states(state, (0, 1)):
                return 0.8
            elif next_state == add_states(state, (0, -1)):
                return 0.2
            else:
                return 0.0
        case ActionT.left:
            if next_state == add_states(state, (-1, 0)):
                return 0.6
            elif next_state == add_states(state, (1, 0)):
                return 0.4
            else:
                return 0.0
        case ActionT.right:
            if next_state == add_states(state, (1, 0)):
                return 0.9
            elif next_state == add_states(state, (-1, 0)):
                return 0.1
            else:
                return 0.0


def reward(state: StateT, action: ActionT) -> float:
    match action:
        case ActionT.stay:
            return REWARD if state == GOAL else COST
        case ActionT.up:
            return REWARD if state == add_states(GOAL, (0, 1)) else COST
        case ActionT.down:
            return REWARD if state == add_states(GOAL, (0, -1)) else COST
        case ActionT.left:
            return REWARD if state == add_states(GOAL, (1, 0)) else COST
        case ActionT.right:
            return REWARD if state == add_states(GOAL, (-1, 0)) else COST


V = [0.0 for _ in state_space]

i = 0

while True:
    Vhat = [
        DISCOUNT * max([
            reward(state, action) + sum([
                V[n] * transition(state, action, next_state)
                for n, next_state in enumerate(state_space)
            ])
            for action in ActionT]) 
        for state in state_space]
    
    average_diff = sum([abs(v - vhat) for v, vhat in zip(V, Vhat)]) / len(state_space)

    i += 1

    if average_diff < CUTOFF:
        print(f'Finished value iteration after {i} iterations with an average change of {average_diff} on the last iteration')
        V = Vhat
        break

    if i % 10 == 0:
        print(f'Reached iteration {i} with a current average change of {average_diff}')
    
    V = Vhat

policy: list[ActionT] = []

for state in state_space:

    action_values = [reward(state, action) + sum([
        V[n] * transition(state, action, next_state)
        for n, next_state in enumerate(state_space)
    ])
    for action in ActionT]

    best_action = list(ActionT)[action_values.index(max(action_values))]

    policy.append(best_action)


def print_value(values: list[float]):
    value_grid = np.zeros((HEIGHT, WIDTH))

    for i, value in enumerate(values):
        x, y = state_space[i]
        value_grid[y, x] = value

    print(value_grid)

def print_policy(values: list[ActionT]):
    policy_lines = [[''] * WIDTH for _ in range(HEIGHT)]

    action_strings: dict[ActionT, str] = {
        ActionT.up: ' ^',
        ActionT.down: ' v',
        ActionT.left: ' <',
        ActionT.right: ' >',
        ActionT.stay: ' *',
    }

    for i, value in enumerate(values):
        x, y = state_space[i]

        policy_lines[y][x] = action_strings[value]

    for line in policy_lines:
        print(' '.join(line))

print_value(V)
print_policy(policy)
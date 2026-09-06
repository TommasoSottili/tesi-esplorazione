
import random


def random_strategy(global_map, frontiers, robot_cell):

    # le frontiere ricevute sono già filtrate a monte (in run_exploration):
    # solo quelle raggiungibili dal robot e diverse dalla sua cella attuale
    if not frontiers:
        return None

    return random.choice(frontiers)

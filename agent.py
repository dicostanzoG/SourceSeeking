import numpy as np


class Agent:
    def __init__(self, name, position, graph):
        self.position = position
        self.name = name
        self.positions = []
        self.speed = 0
        self.distance = 0
        graph.add_node(self.name, pos=self.position)

    def get_name(self):
        return self.name

    def get_position(self):
        return self.position

    def set_speed(self, speed):
        self.speed = speed  # calcolata con metodo di apf, antigradiente del pot

    def get_speed(self):
        return self.speed

    def set_position(self, pos, node):
        self.position = pos
        self.positions.append(pos)
        node[1]['pos'] = pos

    def update_pos(self, Ts, node, source_pos, end, n_agents):
        pos = tuple(self.position + Ts * self.speed)
        self.set_position(pos, node)

        if node[0] == n_agents - 1:
            self.distance = np.linalg.norm(np.array(self.position) - source_pos)
            if self.distance <= 0.05:
                end = True
        else:
            pass
        return self.position, end

    def get_path(self):
        return self.positions

    def get_distance(self):
        return self.distance


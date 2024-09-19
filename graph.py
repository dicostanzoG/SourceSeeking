import sys

import networkx as nx
import numpy as np
from matplotlib import tri
import math
import random
from matplotlib import pyplot as plt
from shapely import Point, Polygon

from agent import Agent


class CreateGraph:
    def __init__(self, g, agents_list, num_agents, r, p, simulator, vertices):
        self.g = g
        # self.max_range = 1
        # self.min_range = -0.5
        self.num_agents = num_agents
        self.positions = []  # non c'è quello centrale
        self.r = r
        self.p = p
        self.agents_list = agents_list
        self.sample_points = np.zeros(
            (2, self.num_agents))  # matrice con le posizioni degli agenti, colonna j=agente, righe=xy
        self.simulator = simulator
        self.area = Polygon(vertices)
        self.min_x, self.min_y, self.max_x, self.max_y = self.area.bounds
        self.lim_x = [self.min_x - 0.1, self.max_x + 0.1]
        # self.lim_y = [self.min_y-0.1, self.max_y+0.1]

    def get_source(self):
        x_s, y_s = random.uniform(self.min_x, self.max_x), random.uniform(self.min_y, self.max_y)
        if not self.in_map((x_s, y_s)):
            x_s, y_s = self.get_source()
        self.g.add_node('Source', pos=(x_s, y_s))
        return x_s, y_s

    def generate_source_movement(self, ax, source_pos):
        ax.cla()
        pos = source_pos
        source_pos = list(source_pos)
        delta_x = random.uniform(-0.02, 0.02)
        delta_y = random.uniform(-0.02, 0.02)
        source_pos[0] += delta_x
        source_pos[1] += delta_y
        if not self.in_map(source_pos):
            source_pos = self.generate_source_movement(ax, pos)
        self.update_source_pos(source_pos)
        return source_pos

    def update_source_pos(self, source_pos):
        self.g.nodes['Source']['pos'] = source_pos

    def neighbors_position(self, agent_pos, neighbor):  # sulla circonferenza di raggio r, agent_pos=robot_c
        angle = 2 * math.pi * neighbor / self.num_agents
        x = agent_pos[0] + self.r * math.sin(angle) + random.gauss(0, 0.07)
        y = agent_pos[1] + self.r * math.sin(angle) + random.gauss(0, 0.07)

        if self.in_map((x, y)):
            for i in self.positions:
                if np.linalg.norm(np.array(i) - np.array((x, y))) <= 0.03:
                    self.neighbors_position(agent_pos, neighbor)
        return x, y

    def get_agent_c(self, source_pos):
        x_c, y_c = random.uniform(self.min_x, self.max_x), random.uniform(self.min_y, self.max_y)
        while not self.in_map((x_c, y_c)) or (np.linalg.norm([x_c, y_c] - np.array(source_pos)) <= 0.5):
            x_c, y_c = random.uniform(self.min_x, self.max_x), random.uniform(self.min_y, self.max_y)
        return x_c, y_c

    def generate_agent_graph(self, x_c, y_c):
        angles = {}
        self.num_agents -= 1
        # plot_circle((x_c, y_c), r)
        # aggiungi nodi attorno al nodo centrale
        for i in range(0, self.num_agents):
            x, y = self.neighbors_position((x_c, y_c), i)
            self.positions.append((x, y))
            d_x = x - x_c
            d_y = y - y_c
            angles[i] = (math.atan2(d_y, d_x))
        return angles

    def generate_graph(self, angles, x_c, y_c):
        robot_c = Agent(self.num_agents, (x_c, y_c), self.g)
        self.agents_list.insert(self.num_agents, robot_c)
        self.sample_points[0, self.num_agents] = x_c
        self.sample_points[1, self.num_agents] = y_c
        angles_sorted = sorted(angles.values())
        i = 0
        for j in angles_sorted:
            for k, val in angles.items():
                if val == j:
                    x, y = self.positions[k]

                    robot_i = Agent(i, (x, y), self.g)
                    self.agents_list.insert(i, robot_i)

                    self.sample_points[0, i] = x
                    self.sample_points[1, i] = y
                    if i != 0:
                        self.g.add_edge(i, i - 1)
                    self.g.add_edge(self.num_agents, i)
                    self.g.add_edge(0, self.num_agents - 1)
                    i += 1
        return self.g, self.sample_points

    def update_sample_points(self, agent_name, x, y):
        self.sample_points[0, agent_name] = x
        self.sample_points[1, agent_name] = y
        return self.sample_points

    def plot_graph(self, ax, i, elements, concentration, method, n_experiment, total_paths,
                   source_positions):
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_xlim(self.lim_x)
        # ax.set_ylim(self.lim_y)

        self.get_obs(ax, elements, concentration)

        if i != 'Total Path':
            color_map = ['red' if j == 'Source' else 'lightblue' for j in self.g.nodes()]
            nodes_g = nx.get_node_attributes(self.g, 'pos')
            nx.draw_networkx(self.g, pos=nodes_g, ax=ax, nodelist=list(nodes_g.keys()), with_labels=True,
                             node_size=6, node_color=color_map, node_shape='s')
        else:
            x = [j[0] for j in total_paths]
            y = [j[1] for j in total_paths]
            ax.plot(x, y, 'o', color='lightblue')

            x_s = [j[0] for j in source_positions]
            y_s = [j[1] for j in source_positions]
            ax.plot(x_s, y_s, 'o', color='red', label='source')

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_title(method)

        filename = f'img/{method}_{n_experiment}_img_{i}.png'
        plt.savefig(filename)
        ax.cla()
        return ax

    def get_obs(self, ax, elements, concentrations):
        for el in elements:
            ax.add_patch(el)
        xy = self.simulator.xy
        ele = self.simulator.ele
        triangles = ele[:3, :]

        triang = tri.Triangulation(xy[0, :], xy[1, :], triangles=triangles.T)
        plt.tripcolor(triang, concentrations.flatten(),
                      cmap='cool',
                      shading='gouraud')  # cool colorazione triangoli basato sulla concentrazione, Wistia, winter, bwr

        # plt.colorbar(label='Concentration', orientation="horizontal")
        ax.add_patch(self.p)  # colora di rosso i bordi del contorno

    def in_map(self, point):
        buffered_polygon = self.area.buffer(-0.3)
        point = Point((point[0], point[1]))
        IN = buffered_polygon.contains(point)
        ON = buffered_polygon.touches(point)
        return IN and not ON

    def update_graph(self, positions, source_pos):
        self.g = nx.Graph()
        name = 0
        self.g.add_node('Source', pos=(source_pos[0], source_pos[1]))
        for i in positions:
            self.g.add_node(name, pos=(i[0], i[1]))
            name += 1
        return self.g

    def set_agents_list(self, value):
        self.agents_list = value
        return self.agents_list


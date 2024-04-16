import networkx as nx
import numpy as np
from matplotlib import tri
import math
import random
from matplotlib import pyplot as plt
from shapely import Point, Polygon
from agent import Agent
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap

class CreateGraph:
    def __init__(self, g, agents_list, num_agents, r, ax, p, simulator):
        self.g = g
        self.max_range = 1
        self.min_range = -0.5
        self.lim_x = [-1.4, 1.4]
        self.lim_y = [-1, 1]
        self.num_agents = num_agents
        self.agents_list = agents_list
        self.positions = []  # non c'è quello centrale
        self.r = r
        self.ax = ax
        self.p = p
        self.sample_points = np.zeros(
            (2, self.num_agents))  # matrice con le posizioni degli agenti, colonna j=agente, righe=xy
        self.simulator = simulator

    def get_source(self, circles):
        x_s, y_s = random.uniform(self.min_range, self.max_range), random.uniform(self.min_range, self.max_range)

        if not self.in_map((x_s, y_s)):
            x_s, y_s = self.get_source(circles)
        self.g.add_node('Source', pos=(x_s, y_s))
        return x_s, y_s

    def generate_source_movement(self, source_pos):
        pos = source_pos
        delta_x = random.uniform(-0.02, 0.02)
        delta_y = random.uniform(-0.02, 0.02)
        source_pos = (source_pos[0] + delta_x, source_pos[1] + delta_y)
        if not self.in_map(source_pos):
            self.g, source_pos = self.generate_source_movement(pos)

        self.g.nodes['Source']['pos'] = (source_pos[0], source_pos[1])
        return self.g, source_pos

    def neighbors_position(self, agent_pos, neighbor, circles):  # sulla circonferenza di raggio r, agent_pos=robot_c
        angle = 2 * math.pi * neighbor / self.num_agents
        x = agent_pos[0] + self.r * math.sin(angle) + random.gauss(0, 0.05)
        y = agent_pos[1] + self.r * math.sin(angle) + random.gauss(0, 0.05)

        for i in self.positions:
            if (not self.in_map((x, y)) or np.linalg.norm(np.array(i) - np.array((x, y))) <= 0.05
                    or np.linalg.norm(np.array(i) - np.array(agent_pos)) <= 0.05):
                x, y = self.neighbors_position(agent_pos, neighbor, circles)

        return x, y

    def generate_agent_graph(self, circles):
        angles = {}
        x_c, y_c = random.uniform(self.min_range, self.max_range), random.uniform(self.min_range, self.max_range)
        if not self.in_map((x_c, y_c)):
            self.generate_agent_graph(circles)
        self.num_agents -= 1
        # plot_circle((x_c, y_c), r)
        robot_c = Agent(self.num_agents, (x_c, y_c), self.g)
        self.agents_list.insert(self.num_agents, robot_c)
        self.sample_points[0, self.num_agents] = x_c
        self.sample_points[1, self.num_agents] = y_c

        # aggiungi nodi attorno al nodo centrale
        for i in range(0, self.num_agents):
            x, y = self.neighbors_position((x_c, y_c), i, circles)
            self.positions.append((x, y))

            d_x = x - robot_c.get_position()[0]
            d_y = y - robot_c.get_position()[1]
            angles[i] = (math.atan2(d_y, d_x))

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

    def plot_graph(self, i, obstacles, elements, concentration):
        color_map = ['red' if j == 'Source' else 'lightblue' for j in self.g.nodes()]
        # shape = ['o' if i == 'Source' else 's' for i in g.nodes()]
        self.ax.set_axis_on()
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlim(self.lim_x)
        plt.ylim(self.lim_y)

        self.get_obs(obstacles, elements, concentration)

        nodes_g = nx.get_node_attributes(self.g, 'pos')
        nx.draw_networkx(self.g, pos=nodes_g, nodelist=list(nodes_g.keys()), with_labels=True,
                         node_size=10, node_color=color_map, node_shape='s')
        plt.pause(0.001)

        plt.savefig('img_' + str(i) + '.png')
        plt.cla()

    def get_obs(self, obstacles, elements, concentrations):
        for el in elements:
            self.ax.add_patch(el)
        xy = self.simulator.xy
        ele = self.simulator.ele
        triangles = ele[:3, :]

        triang = tri.Triangulation(xy[0, :], xy[1, :], triangles=triangles.T)
        plt.tripcolor(triang, concentrations.flatten(),
                      cmap='winter',
                      shading='gouraud')  # colorazione triangoli basato sulla concentrazione, Wistia, winter, bwr

        self.ax.add_patch(self.p)  # colora di rosso i bordi del contorno

    def in_map(self, point):
        polygon = Polygon((self.p.get_xy()))
        IN = polygon.contains(Point((point[0], point[1])))
        ON = polygon.touches(Point((point[0], point[1])))
        if IN:
            return True
        elif ON:
            return False
        else:
            return False


'''''''''
class CreateGraph:
    def __init__(self, g, num_nodes, r, ax, p):
        self.g = g
        self.max_range = 1
        self.min_range = -0.5
        self.lim_x = [-1.4, 1.4]
        self.lim_y = [-1, 1]
        self.num_nodes = num_nodes
        self.r = r
        self.ax = ax
        self.p = p

    def get_source(self, circles):
        x_s, y_s = random.uniform(self.min_range, self.max_range), random.uniform(self.min_range, self.max_range)
        for i in circles:
            if self.collisions(i, x_s, y_s) or not self.in_map((x_s, y_s)):
                x_s, y_s = self.get_source(circles)
        self.g.add_node('Source', pos=(x_s, y_s))
        return x_s, y_s

    def neighbors_position(self, agent_pos, neighbor, circles):  # sulla circonferenza di raggio r
        angle = 2 * math.pi * neighbor / self.num_nodes
        x = agent_pos[0] + self.r * math.cos(angle) + random.gauss(0, 0.05)  # x_c, y_c
        y = agent_pos[1] + self.r * math.sin(angle) + random.gauss(0, 0.05)
        for i in circles:
            if self.collisions(i, x, y) or not self.in_map((x, y)):
                x, y = self.neighbors_position(agent_pos, neighbor, circles)
        return x, y

    def generate_agent_graph(self, agents, circles):
        x_c, y_c = random.uniform(self.min_range, self.max_range), random.uniform(self.min_range, self.max_range)
        for i in circles:
            if self.collisions(i, x_c, y_c) or not self.in_map((x_c, y_c)):
                self.generate_agent_graph(agents, circles)

        self.num_nodes -= 1
        # plot_circle((x_c, y_c), r)
        robot_c = Agent(self.num_nodes, (x_c, y_c), self.g)
        agents.insert(self.num_nodes, robot_c)
        # aggiungi nodi attorno al nodo centrale
        for i in range(0, self.num_nodes):
            x, y = self.neighbors_position((x_c, y_c), i, circles)
            robot_i = Agent(i, (x, y), self.g)
            agents.insert(i, robot_i)
            if i != 0:
                self.g.add_edge(i, i - 1)
            self.g.add_edge(self.num_nodes, i)
            self.g.add_edge(0, self.num_nodes - 1)

        return self.g

    def plot_circle(self, center):
        # genera un'array di angoli
        angles = np.linspace(0, 2 * np.pi, 100)

        # calcola le coordinate dei punti sulla circonferenza
        x = center[0] + self.r * np.cos(angles)
        y = center[1] + self.r * np.sin(angles)

        plt.plot(x, y, 'b-')
        plt.axis('equal')

    def plot_graph(self, i, obstacles, elements):
        color_map = ['red' if j == 'Source' else 'lightblue' for j in self.g.nodes()]
        # shape = ['o' if i == 'Source' else 's' for i in g.nodes()]
        self.ax.set_axis_on()
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlim(self.lim_x)
        plt.ylim(self.lim_y)

        self.get_obs(obstacles, elements)

        nodes_g = nx.get_node_attributes(self.g, 'pos')
        nx.draw_networkx(self.g, pos=nodes_g, nodelist=list(nodes_g.keys()), with_labels=True,
                         node_size=10, node_color=color_map, node_shape='s')
        plt.pause(0.1)

        # plt.savefig('img_' + str(i) + '.png')
        plt.cla()

    def generate_obstacles(self, boxes):
        obstacles = []
        while len(boxes) < 3:

            a = random.uniform(self.min_range, self.max_range)
            b = random.uniform(self.min_range, self.max_range)
            obs = Circle((a, b), random.uniform(0.05, 0.1), color='silver')
            obstacles.append(obs)
            if not self.in_map((a, b)):
                boxes, obstacles = self.generate_obstacles(boxes)

            if len(boxes) == 0:
                boxes.append(obs.clipbox)
                self.ax.add_patch(obs)
            elif len(boxes) > 0:  # and not obs_collision(obs, obstacles):
                boxes.append(obs.clipbox)
                self.ax.add_patch(obs)
            else:
                pass

        return boxes, obstacles

    def get_obs(self, obstacles, elements):
        for obs in obstacles:
            self.ax.add_patch(obs)
        for el in elements:
            self.ax.add_patch(el)
        self.ax.add_patch(self.p)
        # colora di rosso i bordi del contorno

    def obs_collision(self, c, circles):  # verifica che il nuovo ostacolo non interseca nessuno di quelli esistenti
        for obs in circles:
            distance = np.linalg.norm(np.array(c.get_center()) - np.array(obs.get_center()))
            sum_r = obs.get_radius() + c.get_radius()
            if distance <= sum_r:
                return True
        return False

    def collisions(self, circle, x, y):
        collision = circle.contains_point(self.ax.transData.transform([x, y]))  # se il pnt è nell'ostacolo
        return collision

    def in_map(self, element):
        if self.p.contains_point(self.ax.transData.transform(np.array(element))):
            return True
        else:
            return False
            
            
nomi agenti


class CreateGraph:
    def __init__(self, g, agents_list, num_agents, r, ax, p, simulator):
        self.g = g
        self.max_range = 1
        self.min_range = -0.5
        self.lim_x = [-1.4, 1.4]
        self.lim_y = [-1, 1]
        self.num_agents = num_agents
        self.agents_list = agents_list
        self.positions = []  # non c'è quello centrale
        self.r = r
        self.ax = ax
        self.p = p
        self.sample_points = np.zeros(
            (2, self.num_agents))  # matrice con le posizioni degli agenti, colonna j=agente, righe=xy
        self.simulator = simulator

    def get_source(self, circles):
        x_s, y_s = random.uniform(self.min_range, self.max_range), random.uniform(self.min_range, self.max_range)

        if not self.in_map((x_s, y_s)):
            x_s, y_s = self.get_source(circles)
        self.g.add_node('Source', pos=(x_s, y_s))
        return x_s, y_s

    def generate_source_movement(self, source_pos):
        pos = source_pos
        delta_x = random.uniform(-0.02, 0.02)
        delta_y = random.uniform(-0.02, 0.02)
        source_pos = (source_pos[0] + delta_x, source_pos[1] + delta_y)
        if not self.in_map(source_pos):
            self.g, source_pos = self.generate_source_movement(pos)

        self.g.nodes['Source']['pos'] = (source_pos[0], source_pos[1])
        return self.g, source_pos

    def neighbors_position(self, agent_pos, neighbor, circles):  # sulla circonferenza di raggio r, agent_pos=robot_c
        angle = 2 * math.pi * neighbor / self.num_agents
        x = agent_pos[0] + self.r * math.sin(angle) + random.gauss(0, 0.05)
        y = agent_pos[1] + self.r * math.sin(angle) + random.gauss(0, 0.05)

        for i in self.positions:
            if not self.in_map((x, y)) or np.linalg.norm(np.array(i) - np.array((x, y))) <= 0.05 or np.linalg.norm(np.array(i) - np.array(agent_pos)) <= 0.05:
                x, y = self.neighbors_position(agent_pos, neighbor, circles)
                print(len(self.positions))

        return x, y

    def generate_agent_graph(self, circles):
        angles = {}
        x_c, y_c = random.uniform(self.min_range, self.max_range), random.uniform(self.min_range, self.max_range)
        if not self.in_map((x_c, y_c)):
            self.generate_agent_graph(circles)
        self.num_agents -= 1
        # plot_circle((x_c, y_c), r)
        robot_c = Agent(self.num_agents, (x_c, y_c), self.g)
        self.agents_list.insert(self.num_agents, robot_c)
        self.sample_points[0, self.num_agents] = x_c
        self.sample_points[1, self.num_agents] = y_c

        # aggiungi nodi attorno al nodo centrale
        for i in range(0, self.num_agents):
            x, y = self.neighbors_position((x_c, y_c), i, circles)
            self.positions.append((x, y))

            d_x = x - robot_c.get_position()[0] 
            d_y = y - robot_c.get_position()[1]
            angles[i] = (math.atan2(d_y, d_x))

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

    def plot_graph(self, i, obstacles, elements, concentration):
        color_map = ['red' if j == 'Source' else 'lightblue' for j in self.g.nodes()]
        # shape = ['o' if i == 'Source' else 's' for i in g.nodes()]
        self.ax.set_axis_on()
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlim(self.lim_x)
        plt.ylim(self.lim_y)

        self.get_obs(obstacles, elements, concentration)

        nodes_g = nx.get_node_attributes(self.g, 'pos')
        nx.draw_networkx(self.g, pos=nodes_g, nodelist=list(nodes_g.keys()), with_labels=True,
                         node_size=10, node_color=color_map, node_shape='s')
        plt.pause(0.001)

        # plt.savefig('img_' + str(i) + '.png')
        plt.cla()

    def get_obs(self, obstacles, elements, concentrations):
        for el in elements:
            self.ax.add_patch(el)
        xy = self.simulator.xy
        ele = self.simulator.ele
        triangles = ele[:3, :]

        triang = tri.Triangulation(xy[0, :], xy[1, :], triangles=triangles.T)
        plt.tripcolor(triang, concentrations.flatten(),
                      cmap='winter',
                      shading='gouraud')  # colorazione triangoli basato sulla concentrazione, Wistia, winter, bwr

        self.ax.add_patch(self.p)  # colora di rosso i bordi del contorno

    def in_map(self, element):
        if self.p.contains_point(self.ax.transData.transform(np.array(element))):
            return True
        else:
            return False
'''''

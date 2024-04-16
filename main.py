import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.io import loadmat
from graph import CreateGraph
from artificialPotentialField import APFMotionPlanner
from mesh import open_file
from simulation import Simulation
import time


def main():
    a = 0
    cc = []
    results = {}
    g = nx.Graph()
    r = 0.1
    num_agents = 6
    Ts = 0.01
    end = False
    agents_list = []
    concentrations_list = []  # [[] for _ in range(num_agents)]
    path = []
    total_paths = []
    distance = []
    source_positions = []
    c_obs = []
    circles = []
    time_ = []
    file = loadmat('mesh.mat')
    mesh = file['mesh']

    fig, ax = plt.subplots()
    elements, p = open_file(ax, mesh)
    polygon_vertices = p.get_xy()

    simulator = Simulation(mesh, num_agents)
    createGraph = CreateGraph(g, agents_list, num_agents, r, ax, p, simulator)

    #  c_obs, circles = createGraph.generate_obstacles(c_obs)

    g, sample_points = createGraph.generate_agent_graph(circles)
    source_pos = createGraph.get_source(circles)
    source_positions.append(source_pos)

    motion_planner = APFMotionPlanner(source_pos, r, agents_list)

    concentration = simulator.concentration(source_pos)
    createGraph.plot_graph(0, circles, elements, concentration)

    k = 0
    while not end:
        # g, source_pos = createGraph.generate_source_movement(source_pos)
        # source_positions.append(source_pos)

        for i in (g.nodes.data()):
            agent_name = i[0]
            if agent_name != 'Source':
                agent = agents_list[agent_name]
                agent.set_speed(motion_planner.calculate_control(agent, circles, polygon_vertices, simulator))
                pos, end = agent.update_pos(Ts, i, source_pos, end,
                                            num_agents)  # aggiorna la posizione del robot con una velocità costante

                if motion_planner.formation_checking():
                    sample_points = createGraph.update_sample_points(agent_name, pos[0],
                                                                     pos[1])  # se considero sample_point=pos agenti
                    simulator.concentration_point(sample_points,
                                                  agent_name)  # concentrazione nel punto in cui si trova ogni agente

                path.insert(agent_name, agent.get_path())
                if agent_name == num_agents - 1:
                    distance.append(agent.get_distance())

            concentration = simulator.concentration(source_pos)  # nella mesh partendo dalla pos della sorgente
            cc.append(concentration)
        createGraph.plot_graph(k, circles, elements, concentration)
        k += 1

        # time_.append(k)
        # total_paths.append(path)
        # concentrations_list.append(simulator.get_concentrations_list())
        simulator.get_gradient(agents_list)

    print(np.min(cc), np.max(cc))  # 3.488374704160273e-06, 339.0628885768361 // 4.34305275992172e-07 194.59284352212381

    # results['time'] = time_
    # results['path'] = total_paths
    # results['concentrations'] = concentrations_list
    # results['distance'] = distance
    # results['source_position'] = source_positions  # source_pos

    # with open('results.csv', 'w', newline='') as file_csv:
    #   writer = csv.writer(file_csv)
    #   writer.writerow(results.keys())
    #   writer.writerows(zip(*results.values()))


if __name__ == '__main__':
    main()
    # os.system("ffmpeg -r 8 -i C:/Users/dicos/PycharmProjects/SourceSeekingTesi/img_%d.png -vcodec mpeg4 -y prm_eps_n.mp4")

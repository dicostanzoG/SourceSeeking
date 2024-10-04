import csv
import matplotlib.pyplot as plt
import networkx as nx
import os
from scipy.io import loadmat
from graph import CreateGraph
from artificialPotentialField import APFMotionPlanner
from mesh import open_file
from simulation import Simulation
import time


def n_iter(interrupt, n_iterations, t):
    if interrupt:
        n_iterations.append(-1)
    else:
        n_iterations.append(t)
    return n_iterations


def get_source_pos(createGraph, source_positions, source_pos, t, ax):
    if len(source_positions) == t + 1:
        source_pos = createGraph.generate_source_movement(ax, source_pos)
        source_positions.append(source_pos)
    else:
        source_pos = source_positions[t + 1]
        createGraph.update_source_pos(source_pos)
    return source_pos


plt.rcParams['animation.ffmpeg_path'] = ('C:/Users/dicos/Downloads/ffmpeg-master-latest-win64-gpl/ffmpeg-master'
                                         '-latest-win64-gpl/bin/ffmpeg.exe')
ffmpeg_path = 'C:/Users/dicos/Downloads/ffmpeg-master-latest-win64-gpl/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe'
os.environ['PATH'] += os.pathsep + os.path.dirname(ffmpeg_path)
output_folder = "C:/Users/dicos/PycharmProjects/SourceSeekingTesi/video/"

output_folder_csv = "C:/Users/dicos/PycharmProjects/SourceSeekingTesi/csv/"


def main():
    g = nx.Graph()
    n_experiments = 100
    r = 0.1
    num_agents = 7
    Ts = 0.01
    time_limit = 200
    agents_list = []
    concentrations_list = []
    mc_n_iter = []
    wm_n_iter = []
    gm_n_iter = []
    dist_m = []  # con ogni lista associata ad un metodo
    methods = ['Max_concentration', 'Weighted_average', 'Gradient']

    n_interrupted_mc = 0
    n_interrupted_wa = 0
    n_interrupted_grad = 0

    results = {}

    file = loadmat('mesh.mat')
    mesh = file['mesh']

    for h in range(n_experiments):
        print('\n ESPERIMENTO ' + str(h + 1))

        fig, ax = plt.subplots()
        elements, p, polygon_vertices = open_file(ax, mesh)
        source_positions = []

        simulator = Simulation(mesh, num_agents)
        createGraph = CreateGraph(g, agents_list, num_agents, r, p, simulator, polygon_vertices)

        source_pos_init = createGraph.get_source()
        source_positions.append(source_pos_init)

        x_c, y_c = createGraph.get_agent_c(source_pos_init)
        angles = createGraph.generate_agent_graph(x_c,
                                                  y_c)  # genera i nodi intorno a quello centrale e calcola l'angolo
        for method in methods:
            print('\n-' + method)
            source_pos = source_pos_init
            start = time.time()

            distance = []
            time_ = []
            total_paths = []

            end = False
            interrupt = False

            concentration = simulator.concentration(source_pos)
            createGraph.plot_graph(ax, 0, elements, concentration, method, h, total_paths, source_positions)

            g, sample_points = createGraph.generate_graph(angles, x_c, y_c)
            motion_planner = APFMotionPlanner(source_pos, r, agents_list)

            t = 0  # istanti di t, tempo impiegato per raggiungere la sorgente
            n_interrupted = 0

            while not end:
                time_.append(t)
                # source_pos = get_source_pos(createGraph, source_positions, source_pos, t, ax)

                for i in g.nodes.data():
                    agent_name = i[0]
                    if agent_name != 'Source':
                        agent = agents_list[agent_name]
                        agent.set_speed(
                            motion_planner.calculate_control(agent, polygon_vertices, simulator, method))
                        pos, end = agent.update_pos(Ts, i, source_pos, end, num_agents)

                        if motion_planner.formation_checking():
                            sample_points = createGraph.update_sample_points(agent_name, pos[0], pos[1])
                            simulator.concentration_point(sample_points, agent_name)

                        if agent_name == num_agents - 1:
                            distance.append(agent.get_distance())
                            total_paths.append(pos)

                concentration = simulator.concentration(source_pos)
                concentrations_list.append(concentration)
                createGraph.plot_graph(ax, t, elements, concentration, method, h, total_paths, source_positions)

                t += 1

                if t >= time_limit:
                    end = True
                    interrupt = True
                    n_interrupted += 1
                    print('INTERROTTO')

            end_time = time.time()

            # dist_m.insert(methods.index(method), distance)
            plt.cla()
            plt.title('Distanza dalla sorgente: ' + method)
            plt.plot(time_, distance, color='black')
            plt.grid()
            plt.xlabel('Tempo')
            plt.ylabel('Distanza')
            plt.savefig('img/Distanza media dalla Sorgente_' + method + str(h) + '.png')

            if method == methods[0]:
                mc_n_iter = n_iter(interrupt, mc_n_iter, t)
                n_interrupted_mc += n_interrupted
            elif method == methods[1]:
                wm_n_iter = n_iter(interrupt, wm_n_iter, t)
                n_interrupted_wa += n_interrupted
            elif method == methods[2]:
                gm_n_iter = n_iter(interrupt, gm_n_iter, t)
                n_interrupted_grad += n_interrupted

            agents_list = createGraph.set_agents_list([])
            simulator.set_x_k()
            print('tempo impiegato: ', t, 'tempo reale: ', end_time - start)

            plt.cla()
            createGraph.plot_graph(ax, 'Total Path', elements, concentration, method, h, total_paths, source_positions)

            # crea video
            input_pattern = f"C:/Users/dicos/PycharmProjects/SourceSeekingTesi/img/{method}_{h}_img_%d.png"
            output_file = f"{output_folder}SourceSeeking_{method}_{h}.mp4"
            command = f"ffmpeg -r 8 -i {input_pattern} -vcodec mpeg4 -y {output_file} > NUL 2>&1"
            os.system(command)

            # cancella immagini dopo aver creato il video
            for i in range(t):
                img_filename = f'img/{method}_{h}_img_{i}.png'
                os.remove(img_filename)

            # crea file csv
            results['time'] = time_
            results['path'] = total_paths
            results['concentrations'] = concentrations_list
            results['distance'] = distance
            results['source_position'] = source_positions  # source_pos

            csv_file = os.path.join(output_folder_csv, f"{method}_{h}_results.csv")

            with open(csv_file, 'w', newline='') as file_csv:
                writer = csv.writer(file_csv)
                writer.writerow(results.keys())
                writer.writerows(zip(*results.values()))

        plt.close(fig)

    percentage_mc = (n_interrupted_mc / n_experiments) * 100
    percentage_wa = (n_interrupted_wa / n_experiments) * 100
    percentage_grad = (n_interrupted_grad / n_experiments) * 100

    results['percentage_mc'] = [percentage_mc]
    print('Percentuale di esperimenti interrotti: ' + str(percentage_mc) + '%')

    results['percentage_wa'] = [percentage_wa]
    print('Percentuale di esperimenti interrotti: ' + str(percentage_wa) + '%')

    results['percentage_grad'] = [percentage_grad]
    print('Percentuale di esperimenti interrotti: ' + str(percentage_grad) + '%')

    '''''''''''
    plt.cla()
    plt.title('Distanza media dalla sorgente')
    plt.plot(range(len(dist_m[0])), dist_m[0], label=methods[0])
    plt.plot(range(len(dist_m[1])), dist_m[1], label=methods[1])
    plt.plot(range(len(dist_m[2])), dist_m[2], label=methods[2])
    plt.legend()
    plt.xlabel('Tempo')
    plt.ylabel('Distanza')
    plt.savefig('img/Distanza media dalla Sorgente_' + '.png')

    plt.cla()
    plt.title('Distanza dalla sorgente' + methods[0])
    plt.plot(range(len(dist_m[0])), dist_m[0], label=methods[0])
    plt.legend()
    plt.xlabel('Tempo')
    plt.ylabel('Distanza')
    plt.savefig('img/Distanza media dalla Sorgente_' + methods[0] + '_.png')

    plt.cla()
    plt.title('Distanza dalla sorgente' + methods[1])
    plt.plot(range(len(dist_m[1])), dist_m[1], label=methods[1])
    plt.legend()
    plt.xlabel('Tempo')
    plt.ylabel('Distanza')
    plt.savefig('img/Distanza media dalla Sorgente_' + methods[1] + '_.png')

    plt.cla()
    plt.title('Distanza dalla sorgente' + methods[2])
    plt.plot(len(dist_m[2]), dist_m[2], label=methods[2])
    plt.legend()
    plt.xlabel('Tempo')
    plt.ylabel('Distanza')
    plt.savefig('img/Distanza media dalla Sorgente_' + methods[2] + '_.png')
    '''''
    # crea istogramma
    mc_n_iter = [x for x in mc_n_iter if x != -1]
    wm_n_iter = [x for x in wm_n_iter if x != -1]
    gm_n_iter = [x for x in gm_n_iter if x != -1]

    text = [('Max_concentration \n interrupted: ' + str(percentage_mc) + '%'),
            ('Weighted_average \n interrupted: ' + str(percentage_wa) + '%'),
            ('Gradient \n interrupted: ' + str(percentage_grad) + '%')]
    plt.cla()
    plt.hist([mc_n_iter, wm_n_iter, gm_n_iter], edgecolor='black', label=text)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title('Distribuzione dei Tempi per Raggiungere la Sorgente')
    plt.xlabel('Tempo')
    plt.ylabel('Numero di Esperimenti')
    plt.savefig('img/Hist_tot_' + '.png', bbox_inches='tight')


if __name__ == '__main__':
    main()

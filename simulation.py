import random
import numpy as np
from shapely import Point, Polygon


class Simulation:
    def __init__(self, mesh, n_agents):

        self.xy = mesh['xy'][0][0]
        self.ele = mesh['ele'][0][0] - 1
        self.shape = mesh['shape'][0][0]  # per calcolare phi e C, per ogni elemento, colonna j=vertice, interpolare
        self.stiffM = mesh['StiffM'][0][0]
        self.massM = mesh['MassM'][0][0]
        self.deltaT = 0.1
        self.time_steps = 10
        self.u_k = 20  # 5, 10,... intensità sorgente
        #self.num_steps = 5
        self.n_vertices = len(self.xy[0])
        self.concentration_list = np.empty(n_agents)
        self.direction = 0
        self.noise = random.gauss(0, 30)

        self.A_matrix = np.linalg.solve(self.massM + self.deltaT * self.stiffM,
                                        self.massM)  # np.linalg.inv(self.massM) * (self.massM + self.deltaT * self.stiffM) #
        self.B_matrix = np.linalg.inv(self.massM + self.deltaT * self.stiffM) * self.deltaT

        # C = F, vettore riga ma dipende dalla pos dell'agente
        self.F_vector = np.zeros((self.n_vertices,
                                  1))  # dipende dalla pos, tutti zero tranne nell'elemento in cui si trova la sorgente, dim=n_nodi, vettore colonna

        self.x_k = np.zeros((1, self.n_vertices))  # vettore concentrazioni al t k, inizialmente tutto 0

    def concentration(self, source_position):  # nei vertici della mesh
        self.F_vector = self.setFvector(source_position)
        x_k_plus = self.A_matrix.dot(self.x_k.T) + self.B_matrix.dot(self.F_vector).dot(self.u_k)             # self.concentration_matrix.append(self.x_k)
        self.x_k = x_k_plus.T
        return self.x_k

    def setFvector(self, position):  # vettore tutti 0 tranne identificativi vertici del triangolo in cui si trova la sorgente
        for ie in range(self.ele.shape[1]):
            polygon_vertices = self.xy[:, self.ele[0:3, ie]]
            polygon = Polygon((polygon_vertices[:, 0], polygon_vertices[:, 1], polygon_vertices[:, 2]))
            IN = polygon.contains(Point((position[0], position[1])))
            ON = polygon.touches(Point((position[0], position[1])))
            shapeN = self.Get_shapeN_2D_DIFFUSION(ie, position)
            if IN or ON:
                self.F_vector[self.ele[:3, ie][0], :] = shapeN[0, 0]
                self.F_vector[self.ele[:3, ie][1], :] = shapeN[0, 1]
                self.F_vector[self.ele[:3, ie][2], :] = shapeN[0, 2]
                #print('F', shapeN[0, 0], shapeN[0, 1], shapeN[0, 2])
        return self.F_vector

    def get_FEM_functions(self, sample_points):  # seleziona l'elemento in cui si trova l'agente
        spl_vertices = np.zeros((sample_points.shape[1], 3))  # contiene gli indici dei vertici dell'elemento
        spl_shape = np.zeros((sample_points.shape[1], 3))
        for point in range(sample_points.shape[1]):  # per ogni agente verifica in quale elemento si trova
            for ie in range(self.ele.shape[1]):
                polygon_vertices = self.xy[:, self.ele[:3, ie]]
                polygon = Polygon((polygon_vertices[:, 0], polygon_vertices[:, 1], polygon_vertices[:, 2]))
                IN = polygon.contains(Point((sample_points[0, point], sample_points[1, point])))
                ON = polygon.touches(Point((sample_points[0, point], sample_points[1, point])))
                if IN or ON:
                    spl_vertices[point, :] = self.ele[:3, ie]  # vertici dell'elemento in cui si trova ogni agente
                    spl_shape[point, :] = self.Get_shapeN_2D_DIFFUSION(ie, sample_points[:, point])  # ottengo Φ(i)
        return spl_vertices.T, spl_shape.T
                                                                                                                           # spl_shape=Φ(i)

    def Get_shapeN_2D_DIFFUSION(self, element, point):  # ottengo Φ(i), i=vertice data posizione agente
        n_vertices = 3
        shapeN = np.zeros((1, n_vertices))
        for vertex in range(n_vertices):                                                                     # (self.shape[:, element][0][:, vertex])  # [:, element]=3 colonne che rappresentano i 3 vertici, [:, vertex]=vertice selezionato
            shapeN[:, vertex] = np.dot((self.shape[:, element][0][:, vertex]), [1, point[0], point[1]])  # Φ(i)=a+bx+cy

        return shapeN

    def concentration_point(self, sample_points, agent):
        spl_vertices, spl_shape = self.get_FEM_functions(sample_points)
        x = 0
        for i in range(3):  # num riga
            node = spl_vertices[i, agent]  # vertice i dell'elemento in cui si trova l'agente
            x += spl_shape[i, agent] * self.x_k[0][int(node)]

        #print("Concentrazione:", x, "rilevata dall'agente", agent)
        #print(x, self.noise)
        self.concentration_list[agent] = x + self.noise

    def max_concentration(self, agent, agents_list):
        if agent.get_name() == len(agents_list) - 1:  # se è l'agente centrale
            index = np.argmax(self.concentration_list)  # agente che ha la concentrazione max
            if index == agent.get_name():
                return 0
            else:
                agent_b = agents_list[index]  # agente con concentrazione max
                dir = 1 * (np.array(agent_b.get_position(), dtype=np.float64) - np.array(agent.get_position(),
                                                                                          dtype=np.float64))
                return dir
        else:
            return 0

    def get_concentrations_list(self):
        return self.concentration_list

    def media_pesata_delle_direzioni(self, agents_list):
        x_min = np.min(self.concentration_list)
        sum_concentrations = 0
        agent_c = agents_list[len(agents_list)-1]
        self.direction = 0
        for i in self.concentration_list:
            sum_concentrations += (i - x_min)
        for j in range(len(agents_list)-1):
            agent = agents_list[j]
            w_i = (self.concentration_list[agent.get_name()] - x_min) / sum_concentrations
            self.direction += w_i * (np.array(agent.get_position()) - np.array(agent_c.get_position()))
        return self.direction

    def get_gradient(self, agents_list):
        M = []  # matrice delle differenze delle posizioni con agente centrale
        v = []  # matrice delle differenze delle concentrazioni con agente centrale
        agent_c = agents_list[len(agents_list)-1]

        for i in range(len(agents_list)-1):
            agent = agents_list[i]
            M.insert(i, (np.array(agent.get_position()) - np.array(agent_c.get_position())).T)
            v.insert(i, (self.concentration_list[i]-self.concentration_list[agent_c.get_name()]))

        v = np.array(v).T
        M = np.array(M)
        grad_c = np.linalg.lstsq(M, v, rcond=None)[0] #Return the least-squares solution to a linear matrix equation.
        return grad_c

    def set_x_k(self):
        self.x_k = np.zeros((1, self.n_vertices))




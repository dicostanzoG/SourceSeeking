import math
import numpy as np


class APFMotionPlanner:
    def __init__(self, source_pos, r, agents_list):
        self.source_pos = np.array(source_pos)  # pos sorgente
        self.k_att = 1  # coefficiente attrattivo
        self.r = r
        self.k_rep_agents = 0.08
        self.k_rep_central = 0.05
        self.k_rep_obs = 0.1
        self.k_pol = 0.1  # perimetro
        self.agents_list = agents_list
        self.a = self.r * math.cos(math.pi / (len(self.agents_list) - 1))  # apotema
        self.l = 2 * math.sqrt(self.r ** 2 - self.a ** 2)  # lato poligono
        self.delta = 0.05
        self.d_map = 0.07

    def attractive_potential(self, agent):
        # forza attrattiva verso sorgente
        d = np.linalg.norm(self.source_pos - np.array(agent.get_position()))
        pot = 0.5 * self.k_att * d ** 2
        grad = self.k_att * (np.array(self.source_pos, dtype=np.float64) -
                             np.array(agent.get_position(), dtype=np.float64))
        return 0  # DA RITORNARE GRAD

    def formation_potential(self, agent):  # tra agenti e centro
        force = (np.array([0, 0])).astype(np.float64)
        agent_pos = np.array(agent.get_position())
        agent_c = self.agents_list[len(self.agents_list) - 1]
        if agent_c.get_name() != agent.get_name():
            agent_c_pos = np.array(agent_c.get_position())
            # calcola distanza euclidea tra robot corrente e altri robot
            d = np.linalg.norm(agent_pos - agent_c_pos)
            # calcola forza repulsiva
            if d != self.r and d != 0:
                force = self.k_rep_central * ((1 / d) - (1 / self.r)) * (agent_pos - agent_c_pos) / d
        else:
            for i in self.agents_list:
                i_pos = np.array(i.get_position())
                d = np.linalg.norm(agent_pos - i_pos)
                # calcola forza repulsiva
                if d != self.r and d != 0:
                    force += self.k_rep_central * ((1 / d) - (1 / self.r)) * (agent_pos - i_pos) / d
        return force

    def agents_potential(self, agent):  # tra agenti, per formazione
        force = (np.array([0, 0])).astype(np.float64)
        agent_pos = np.array(agent.get_position())
        n = len(self.agents_list)
        for agent_b in self.agents_list:
            if (agent.get_name() != n - 1 and agent_b.get_name() != n - 1) and (agent.get_name() != agent_b.get_name()):
                agent_b_pos = np.array(agent_b.get_position())
                if ((agent.get_name() != n - 2 and agent_b.get_name() == agent.get_name() + 1) or  # tra agenti vicini
                        (agent.get_name() != 0 and agent_b.get_name() == agent.get_name() - 1) or
                        (agent.get_name() == 0 and agent_b.get_name() == n - 2) or
                        (agent.get_name() == n - 2 and agent_b.get_name() == 0)):
                    d = np.linalg.norm(agent_pos - agent_b_pos)
                    # calcola forza repulsiva
                    if d != self.l and d != 0:
                        force += self.k_rep_agents * ((1 / d) - (1 / self.l)) * (agent_pos - agent_b_pos) / d
                else:
                    d = np.linalg.norm(agent_pos - agent_b_pos)
                    if d < 2 * self.r and d != 0:
                        force += self.k_rep_agents * ((1 / d) - (1 / (2 * self.r))) * (agent_pos - agent_b_pos) / d

        return force

    def repulsive_pot_map(self, agent, polygon_vertices):
        tot_force = (np.array([0, 0])).astype(np.float64)
        for i in polygon_vertices:
            distance = np.linalg.norm(np.array(i) - agent.get_position())
            if distance <= self.d_map:
                tot_force += (self.k_pol * ((1 / distance) - (1 / self.d_map)) *
                              (np.array(agent.get_position()) - np.array(
                                  i)) / distance)  # np.linalg.norm(obstacle_position - agent_pos)
            else:
                tot_force += 0

        return tot_force

    def total_potential(self, agent, polygon_vertices, simulator, method):
        m = 0
        force = (self.formation_potential(agent) + self.repulsive_pot_map(agent, polygon_vertices))
        if agent.get_name() == len(self.agents_list) - 1:  # se è l'agente centrale ha pot att
            if self.formation_checking():
                if method == 'Max_concentration':
                    m = simulator.max_concentration(agent, self.agents_list)
                elif method == 'Weighted_average':
                    m = simulator.media_pesata_delle_direzioni(self.agents_list)
                elif method == 'Gradient':
                    m = simulator.get_gradient(self.agents_list)
                else:
                    print('ERROR')
                return force + m
            else:
                return force
        else:  # gli agenti sulla circonferenza non hanno pot att verso la sorgente ma quello repulsivo da i loro vicini
            return force + self.agents_potential(agent)

    def calculate_control(self, agent, polygon_vertices, simulator, method):
        control = self.total_potential(agent, polygon_vertices, simulator, method)
        # normalizzazione vettore di controllo
        control_norm = np.linalg.norm(control)
        if control_norm > 0:
            control = control / control_norm
        else:
            control = np.array([0, 0])
        return control

    def formation_checking(self):
        n = len(self.agents_list)
        formation = False
        d_r = []
        d_l = []
        for i in range(n):
            current_agent = self.agents_list[i]
            if current_agent.get_name() == n - 1:  # se è quello centrale, calcola il raggio
                for j in range(n - 1):
                    next_agent = self.agents_list[j]  # agenti sulla circonferenza
                    d_r.append(
                        np.linalg.norm(np.array(current_agent.get_position()) - np.array(next_agent.get_position())))
            else:
                next_agent = self.agents_list[i + 1] if i < (len(self.agents_list) - 2) else self.agents_list[0]
                d_l.append(np.linalg.norm(np.array(current_agent.get_position()) - np.array(next_agent.get_position())))
        r_formation = all((self.r - 0.02 <= r <= self.r + 0.02) for r in d_r)
        l_formation = all((self.l - 0.02 <= h <= self.l + 0.02) for h in d_l)

        if r_formation and l_formation:
            formation = True

        return formation


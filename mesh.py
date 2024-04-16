import numpy as np
from matplotlib.patches import Polygon


def open_file(ax, mesh):
    elements = []

    xy = mesh['xy'][0][0]
    ele = mesh['ele'][0][0]

    for j in range(ele.shape[1]):  # colonna j
        triangle = ele[:3, j]

        node1 = xy[:, int(triangle[0] - 1)]
        node2 = xy[:, int(triangle[1] - 1)]
        node3 = xy[:, int(triangle[2] - 1)]

        polygon = Polygon([node1, node2, node3], closed=True, edgecolor='gray', fill=False)
        elements.append(polygon)
        ax.add_patch(polygon)

    p = find_outer_edges(ele, xy, ax)  # limite mesh
    ax.add_patch(p)
    return elements, p


def find_outer_edges(mesh_elements, mesh_vertices, ax):
    # trova tutti i lati (coppie di vertici) dei triangoli
    edges = set()
    edge_counts = {}

    for j in range(mesh_elements.shape[1]):
        triangle = mesh_elements[:3, j]
        for i in range(3):
            edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3]))) # %3 calcola l'indice del prossimo vertice nel triangolo
            edges.add(edge)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    # trova i lati esterni che compaiono solo una volta
    outer_edges = [edge for edge, count in edge_counts.items() if count == 1]
    polygon_vertices = []
    for edge in outer_edges:
        vertex1 = mesh_vertices[:, int(edge[0] - 1)]
        vertex2 = mesh_vertices[:, int(edge[1] - 1)]
        #ax.plot([vertex1[0], vertex2[0]], [vertex1[1], vertex2[1]], color='red')
        if vertex1.tolist() not in polygon_vertices:
            polygon_vertices.append(vertex1.tolist())
        if vertex2.tolist() not in polygon_vertices:
            polygon_vertices.append(vertex2.tolist())

    # calcola centroide del poligono per determinare ordinamento
    centroid = np.mean(polygon_vertices, axis=0)

    # ordinamento vertici in base all'angolo rispetto al centroide
    polygon_vertices.sort(key=lambda v: np.arctan2(v[1] - centroid[1], v[0] - centroid[0]))

    p = Polygon(polygon_vertices, closed=True, edgecolor='red', fill=False)
    return p


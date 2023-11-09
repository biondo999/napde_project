import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
# from scipy.special import roots_legendre
# import scipy.integrate
import quad
from GaussJacobiQuadRule_V3 import *

# TODO: implement B

# TODO: implement I
# TODO: implement all plots:
    # H1 error
    # Integration convergence
    # IVPINN v VPINN

# TODO: change area input to length
# TODO: experiment with performance difference when doing smthn like scipy integrate over each triangle instead


print("mesh_lib imported")


class Mesh:
    def __init__(self, domain: list, H: float = None, N: float = None, mesh: dict = None) -> None:
        if domain:
            self.domain = self.create_domain(domain)
            if H:
                self.generate_mesh(H)

            if H and N:
                self.generate_full_mesh(H, N)
        else:
            self.domain = []
            self.mesh = mesh 
            self.N = self._get_number_elements()


    def generate_mesh(self, h: float) -> None:
        """Generates Delaunay mesh of given domain."""
        self.mesh = self.create_mesh(self.domain, h)
        self.N = self._get_number_elements()



    @staticmethod
    def create_domain(domain: list) -> dict:
        """Helper method to simplify the definition of a domain. Triangle takes
        domain vertices via a dictionary"""
        return dict(vertices=np.array(domain))

    @staticmethod
    def create_mesh(domain: dict, h: float):
        """Wraps the Triangle methods that performs the triangularisation."""
        # N.B. e flags the addition of edges to mesh properties, q ensures no element angles 
        # of sub 20deg, and a permits the inclusion of an area constraint, h.
        return tr.triangulate(domain, f"eqa{h}")

    def _get_vertices(self):
        """Returns the coordinates of all points in the mesh."""
        return self.mesh["vertices"]

    def _get_edges(self):
        """Returns the node pairs for all edges in the mesh."""
        return self.mesh["edges"]

    def _get_number_elements(self):
        """Returns the number of elements in the mesh (= number of triangles)."""
        return len(self._get_elements())

    def _get_elements(self):
        """Returns the node trios that create the triangles"""
        return self.mesh["triangles"]

    def _get_element_points(self, n: int):
       """
       Returns the vertices of an element of the mesh
       """
       return self._get_vertices()[self._get_elements()[n]]

    @staticmethod
    def specify_edge_length(h: float):
        """Allows the definition of element size via max. edge length, instead
        of area."""
        #TODO: DOES NOT WORK
        return h * h / 3

    def generate_sub_mesh(self, element: int, N: int):
        """Creates """
        # TODO: extend to a selection of multiple elements

        sub_mesh = self.generate_reference_sub_mesh(N)
        sub_mesh['vertices'], J = self.translate(sub_mesh['vertices'], self._get_element_points(element))
        return Mesh(0, mesh=sub_mesh)

    def compare(self, other = None):
        if other:
            tr.compare(plt, self.mesh, other.mesh)
        else:
            tr.compare(plt, self.domain, self.mesh)
        plt.show()

    def generate_full_mesh(self, H: float, N: float):
        self.meshed_elements = []
        self.generate_mesh(H)
        sub_mesh = self.generate_reference_sub_mesh(N)

        for element in range(self.N):
            self.meshed_elements.append(self.generate_sub_mesh(element, N))

    def plot_sub_mesh(self, to_plot: list = [], figsize: tuple = (6, 6)):
        """
        Plots the sub meshes for the indicated elements, can be one element or all
        """

        plt.figure(figsize=figsize)

        plt.plot(
            self._get_vertices()[:, 0], self._get_vertices()[:, 1], "bo"
        )  # Plot element vertices as blue dots

        if not to_plot:
            to_plot = np.arange(0, self.N, 1)



        for element in to_plot:
            plt.triplot(
                (self.meshed_elements[element])._get_vertices()[:, 0],
                (self.meshed_elements[element])._get_vertices()[:, 1],
                (self.meshed_elements[element])._get_elements(),
                color="lightgreen",
            )

        plt.triplot(
            self._get_vertices()[:, 0],
            self._get_vertices()[:, 1],
            self._get_elements(),
        )

        plt.plot(
            self.domain["vertices"][:, 0], self.domain["vertices"][:, 1], "ro"
        )  # Plot original vertices as red dots

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def _get_edge_lengths(self):
        edge_points = self._get_vertices()[self._get_edges()]
        edge_points = np.array([[edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]] for edge in edge_points])
        return np.sqrt((edge_points[:, 0] ** 2) + (edge_points[:, 1] ** 2))

    def max_H(self):
        return max(self._get_edge_lengths())

    def min_H(self):
        return min(self._get_edge_lengths())

    def min_h(self):
        return min(map(lambda sub_mesh: sub_mesh.min_H(), self.meshed_elements))

    def get_refinement_ratio(self):
        return self.min_h() / self.max_H()

    @staticmethod
    def translate(nodes: list, vertices: list):
        #todo maybe change vertices from list to np.array of type np.float64
        """
        Translates given nodes in the reference
        case ([0, 0], [1, 0], [0, 1]) to an arbritrary triangle
        """

        # print(nodes)
        # print(vertices)
        #convert this so everything works with numpy arrays



        output = np.zeros(shape=np.shape(nodes))

        output[:, 0] = (
            nodes[:, 0] * (vertices[1, 0] - vertices[0, 0])
            + nodes[:, 1] * (vertices[2, 0] - vertices[0, 0])
            + vertices[0, 0]
        )

        output[:, 1] = (
            nodes[:, 0] * (vertices[1, 1] - vertices[0, 1])
            + nodes[:, 1] * (vertices[2, 1] - vertices[0, 1])
            + vertices[0, 1]
        )

        det = (vertices[1, 0] - vertices[0, 0]) * (vertices[2, 1] - vertices[0, 1]) - (
            vertices[2, 0] - vertices[0, 0]
        ) * (vertices[1, 1] - vertices[0, 1])

        return output, det
    
    def get_refinement_ratio(self):
        return self.min_h() / self.max_H()

    @staticmethod
    def translate_der(vertices: list,dx:np.array,dy:np.array):
        """
        vertices:list of vertices where you want to tralste into 
        dx:eval of the der wrt x of the basis on certian points 
        dy:eval of the der wrt x of the basis on certian points 

        traslate eval of ref basis of triangle into a proper eval of the basis of traslated triangle   

        -->traslated dx,traslated dy,jac 
        """

        outputx = np.zeros(shape=(np.shape(dx)),dtype=np.float64)
        outputy = np.zeros(shape=(np.shape(dy)),dtype=np.float64)

        # a= (vertices[1, 0] - vertices[0, 0])
        # b= (vertices[2, 0] - vertices[0, 0])
        # c= (vertices[1, 1] - vertices[0, 1])
        # d= (vertices[2, 1] - vertices[0, 1])

        #   B=[[a,b],
        #      [c,d]]
        #    
        #   B^-T=[[d,-c],
        #      [-b,a]]
        #



        det = (vertices[1, 0] - vertices[0, 0]) * (vertices[2, 1] - vertices[0, 1]) - (
            vertices[2, 0] - vertices[0, 0]
        ) * (vertices[1, 1] - vertices[0, 1])        



        outputx = (
            dx * (vertices[2, 1] - vertices[0, 1])
            - dy * (vertices[1, 1] - vertices[0, 1])
        )/det

        outputy = (
            -dx * (vertices[2, 0] - vertices[0, 0])
            + dy * (vertices[1, 0] - vertices[0, 0])
        )/det



        return outputx,outputy,det        


    def print_info(self):
        print('mesh info : ')
        print('  number of elements = ',self._get_number_elements())
        print('  H_max = ',self.max_H())
        print('  H_min = ',self.min_H())
        print('  refinement ratio = ',self.get_refinement_ratio())
        print('submesh info : ')
        print('  number of elements = ',self.meshed_elements[0]._get_number_elements())
        print('  h_min = ',self.min_h())
        #print('  refinement ratio = ',self.meshed_elements[0].get_refinement_ratio())


    @staticmethod
    def GLQ():
        """
        Generate the Gauss-Legendre nodes and weights for
        numerical integration
        """
        return np.array(quad.points), np.array(quad.weights).transpose()

    # Calculate characteristic size of the mesh (length of longest edge among all triangles)
    def integrate(self, f) -> float:
        I = []
        for element in range(self.N):
            I.append(self.integrate_element(element, f))
        return I, np.sum(I)

    def integrate_element(self, element, f):
        nodes, weights = self.GLQ()
        points = self._get_element_points(element)
        # print(points)
        normalised_points, J = self.translate(nodes, points)
        return J * np.sum(f(normalised_points) * weights) / 2

    def convergence(self, f, exact):
        e = []
        n = []
        for order in range(10, 201, 10):  # Specify the desired order
            points, weights = self.GLQ()

            n.append(order)
            tot = 0.0
            _,tot = self.integrate(f)

            e.append(np.abs(tot - exact))
        e = np.asarray(e)
        n = np.asarray(n)

        plt.figure(figsize=(8, 6))
        plt.loglog(1 / n, e, marker="o", linestyle="-", color="b", label="Error")
        plt.loglog(1 / n, 1 / n, marker="o", linestyle="-", color="g", label="1/n")

        plt.xlabel("1/n")
        plt.title("Convergence Plot")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()

    def generate_reference_sub_mesh(self, N: int):

        x = np.linspace(0, 1, N)
        xx, yy = np.meshgrid(x,x)
        xx = xx.flatten(); yy = yy.flatten()
        filt = xx + yy <= 1
        grid = np.vstack([xx[filt], yy[filt]]).transpose()

        nodes_to_iterate = np.arange(0, (N+1)*N//2)
        nodes_to_drop = np.cumsum(np.arange(N, 0, -1)) - 1
        nodes_to_iterate = [i for i in nodes_to_iterate if i not in nodes_to_drop]

        cells = []
        w = N+1
        for i in nodes_to_iterate:
            if i - 1 in nodes_to_iterate:
                cells.append([i, i+w, i+w-1])
            else:
                w -= 1
            cells.append([i, i+1, i+w])

        sub_mesh = tr.triangulate(dict(vertices=grid))
        sub_mesh['triangles'] = cells

        edges = [[[e[0], e[1]], [e[0], e[2]], [e[1], e[2]]] for e in cells]
        # edges = [np.array(e).flatten().tolist() for e in edges]
        edges = np.reshape(np.array(edges).flatten(), (-1, 2)).tolist()
        sub_mesh['edges'] = edges
        return sub_mesh
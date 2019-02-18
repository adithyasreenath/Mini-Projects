#Imports
from collections import Counter, defaultdict, deque
import copy
from itertools import combinations
import math
import networkx as nx
import urllib.request


#Community Detection

def example_graph():
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def bfs(graph, root, max_depth):
    node2distances = dict()
    visitednode = dict()
    node2num_paths = dict()
    node2parents = dict()
    #depth = 0
    bfsQueue = deque()


    nodes = graph.node
    for node in nodes:
        visitednode[node] = 0

    bfsQueue.append(root)
    visitednode[root] = 1
    node2distances[root] = 0
    node2num_paths[root]=1

    while bfsQueue:
        current = bfsQueue.popleft()
        neighbors = set(graph.neighbors(current))

        for node in neighbors:
            if not visitednode[node] == 1:
                #depth = node2distances[current] + 1
                if (node2distances[current] + 1)<= max_depth:
                #if(depth<=max_depth):
                    bfsQueue.append(node)

                    parents  = set(graph.neighbors(node))
                    parentList = []
                    for parent in parents:
                        if visitednode[parent] == 1:
                            if node2distances[parent] == node2distances[current]:
                                if not parentList.__contains__(parent):
                                    parentList.append(parent)

                    node2parents[node]= parentList
                    node2num_paths[node]= len(parentList)

                    visitednode[node] = 1
                    node2distances[node]= node2distances[current] + 1

    return node2distances,node2num_paths,node2parents
    pass


def complexity_of_bfs(V, E, K):
    complexity_bfs = V + E
    return complexity_bfs
    pass


def bottom_up(root, node2distances, node2num_paths, node2parents):
    sorted_node_list = sorted(node2distances.items(), key=lambda x: x[1], reverse=True)
    edge_credits = {}
    node_credits = {root: 0}
    for node, distance in sorted_node_list:
        if node == root:
            continue
        if not(node in node_credits):
            node_credits[node] = 1
        per_edge_credit = node_credits[node] / node2num_paths[node]
        for parent in node2parents[node]:
            if not (parent in node_credits):
                node_credits[parent] = 1
            node_credits[parent] += per_edge_credit
            edge_credits[tuple(sorted([node, parent]))] = per_edge_credit
#    print("BOTTOM UP START: ")
#    print(sorted(edge_credits.items()))
#    print("BOTTOM UP END: ")
    return edge_credits
    pass


def approximate_betweenness(graph, max_depth):
#   max_depth = 2
    all_edge_betweenness_list = [bottom_up(node, *bfs(graph, node, max_depth)) for node in graph.nodes()]
    all_edge_betweenness = defaultdict(list)
    for edge_dicts in all_edge_betweenness_list:
        [all_edge_betweenness[edge].append(betw) for edge, betw in edge_dicts.items()]
    result = {}
    for edge, betw_list in all_edge_betweenness.items():
        result[edge] = sum(betw_list) / 2
#    print("APPROX BETWEENESS : ", result.items())
    return result
    pass


def get_components(graph):
    return [c for c in nx.connected_component_subgraphs(graph)]

def partition_girvan_newman(graph, max_depth):
    graph_to_partition = graph.copy()
    edgeList = approximate_betweenness(graph_to_partition, max_depth).items()
    result = sorted(edgeList, key=lambda x: -x[1])
    sub_graphs = list()
    for (edgeu, edgev) , betw in result:
        graph_to_partition.remove_edge(edgeu, edgev)
        sub_graphs = list(nx.connected_component_subgraphs(graph_to_partition))
        if len(sub_graphs) > 1:
            break

    return sub_graphs
    pass

def get_subgraph(graph, min_degree):
    degreeList = [node for node, degree in dict(nx.degree(graph)).items() if degree >= min_degree]
    subgraph = graph.subgraph(degreeList)
#    print(sorted(subgraph.nodes()))
#    print(len(subgraph.edges()))    
    return subgraph
    pass

def volume(nodes, graph):
    graph_cpy = graph.copy()
    sub_graph = graph.subgraph(nodes)
    graph_cpy.remove_edges_from(sub_graph.edges())
    count = len(graph_cpy.edges(nodes)) + sub_graph.number_of_edges()
#    print(count)
    return count
    pass


def cut(S, T, graph):
    edges = 0
    for s in S:
        for t in T:
            if graph.has_edge(s,t):
                edges = edges + 1

    return edges
    pass



def norm_cut(S, T, graph):
    Cut = cut(S,T,graph)
    volume_s = volume(S,graph)
    volume_t = volume(T,graph)

    ncv = (Cut/volume_s) + (Cut/volume_t)

    return ncv
    pass


def brute_force_norm_cut(graph, max_size):
    graph_cpy = graph.copy()
    edge_list = graph_cpy.edges()
    first_list = []
    final_list = []
    count = 0
#    print(edge_list)
    for i in range(1, (int)(max_size + 1)):
        comb = [list(x) for x in combinations(edge_list, i)]
        first_list.extend(comb)
#    print("first_LIST", first_list)
    for x in first_list:
        count = 0
        edges = []
        for i in x:
#            i = list(i)
            graph_cpy.remove_edge(i[0],i[1])
            edges.append((i[0], i[1]))
        if(nx.number_connected_components(graph_cpy) == 2):
#            nx.draw_networkx(graph_cpy)
#            plt.show()
            graphs_sub = list(nx.connected_component_subgraphs(graph_cpy))
            scores = norm_cut(graphs_sub[0].nodes,graphs_sub[1].nodes,graph)    
            final_list.append((scores, edges))
#            print("x = ", x,"score = ",scores)
        graph_cpy = graph.copy()
#    print(final_list)
    return final_list
    pass



=======
def brute_force_norm_cut(graph, max_size):
   pass



def score_max_depths(graph, max_depths):
    lst = []
    for depth in max_depths:
        components = list(partition_girvan_newman(graph,depth))
#        print(components[0], components[1])
        value = norm_cut(components[0],components[1],graph)
        lst.append((depth,value))

    return lst
    pass


# Link prediction

def make_training_graph(graph, test_node, n):
    graph_cpy = graph.copy()
    edges = [(test_node, node) for node in sorted(graph.neighbors(test_node))[0:n]]
    graph_cpy.remove_edges_from(edges)
    return graph_cpy
    pass



def jaccard(graph, node, k):
    neighbors = set(graph.neighbors(node))
    scores = []
    for n in graph.nodes():
        if not neighbors.__contains__(n):
            neighbors2 = set(graph.neighbors(n))
            scores.append(((node,n), float(1. * len(neighbors & neighbors2)) / float(len(neighbors | neighbors2))))

    sorted_temp = sorted(scores, key=lambda x: (x[0]))
    temp = sorted(sorted_temp, key=lambda x: (x[1]), reverse=True)
    final = list()
    for i in range(1,k+1):
        final.append(temp[i])

    return  final
    pass



def evaluate(predicted_edges, graph):
    no_present_edges = len([edge for edge in predicted_edges if graph.has_edge(edge[0], edge[1])])

    return no_present_edges/len(predicted_edges)
    pass

def download_data():
    #Download the data


def read_graph():
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')


def main():
    download_data()
    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1,5)))
    clusters = partition_girvan_newman(subgraph, 3)
    print('%d clusters' % len(clusters))
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    print('smaller cluster nodes:')
    print(sorted(clusters, key=lambda x: x.order())[0].nodes())
    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
          (train_graph.order(), train_graph.number_of_edges()))


    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' %
          evaluate([x[0] for x in jaccard_scores], subgraph))


if __name__ == '__main__':
    main()

import networkx as nx

def increment_edge_attribute(graph, node1, node2, att_name, increment):
    if graph.has_edge(node1, node2):
        edge = graph[node1][node2]
        if att_name in edge:
            edge[att_name] += increment
        else:
            edge[att_name] = increment
    else:
        graph.add_edge(node1, node2, **{att_name: increment})

def increment_node_attribute(graph, element, att_name, increment):
    if graph.has_node(element):
        node = graph.node[element]
        if att_name in node:
            node[att_name] += increment
        else:
            node[att_name] = increment


def get_node_att_or_set_default(graph, node, att_name, compute_default):
    """
    If node attribute exists, return it.
    Otherwise compute it using compute_default lambda function, set it to node and return it.
    :param graph:
    :param node:
    :param att_name:
    :param compute_default:
    :return:
    """
    if att_name in node:
        return node[att_name]
    else:
        att_value = compute_default(node)
        node[att_name] = att_value
        return att_value
import numpy as np
import networkx as nx
from io import BytesIO
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from utils.refs import graph_color_ref

def add_text(text, imgarr):
    '''
    Function to add text to image

    Args:
    - text (str): text to add
    - imgarr (np.array): image array

    Returns:
    - img (np.array): image array with text
    '''
    img = Image.fromarray(imgarr)
    I = ImageDraw.Draw(img)
    I.text((10, 10), text, fill='black')
    return np.asarray(img)

def get_color(ref, n_nodes):
    '''
    Function to color the nodes

    Args:
    - ref (list): list of color reference
    - n_nodes (int): number of nodes

    Returns:
    - colors (list): list of colors
    '''
    N = len(ref)
    colors = []
    for i in range(n_nodes):
        colors.append(np.array([[int(i) for i in ref[i%N][4:-1].split(',')]]) / 255.)
    return colors


def make_grid(images, cols=5):
    """
    Arrange list of images into a N x cols grid.
    
    Args:
    - images (list): List of Numpy arrays representing the images.
    - cols (int): Number of columns for the grid.
    
    Returns:
    - grid (numpy array): Numpy array representing the image grid.
    """
    # Determine the dimensions of each image
    img_h, img_w, _ = images[0].shape
    rows = len(images) // cols
    
    # Initialize a blank canvas
    grid = np.zeros((rows * img_h, cols * img_w, 3), dtype=images[0].dtype)
    
    # Place each image onto the grid
    for idx, img in enumerate(images):
        y = (idx // cols) * img_h
        x = (idx % cols) * img_w
        grid[y: y + img_h, x: x + img_w] = img
    
    return grid

def viz_graph(info_dict, res=256):
    '''
    Function to plot the directed graph

    Args:
    - info_dict (dict): output json containing the graph information
    - res (int): resolution of the image

    Returns:
    - img_arr (np.array): image array
    '''
    # build tree
    tree = info_dict['diffuse_tree']
    edges = []
    for node in tree:
        edges += [(node['id'], child) for child in node['children']]
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # plot tree
    plt.figure(figsize=(res/100, res/100))

    colors = get_color(graph_color_ref, len(tree))
    pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
    node_order = sorted(G.nodes())
    nx.draw(G, pos, node_color=colors, nodelist=node_order, edge_color='k', with_labels=False)
    
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    img_arr = np.asarray(img)
    buf.close()
    plt.clf()
    plt.close()
    return img_arr[:, :, :3]

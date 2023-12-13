import os
import json
import networkx as nx

def get_hash(file, key='diffuse_tree', ignore_handles=True, dag=False):
    tree = file[key]
    if dag:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for node in tree:
        if ignore_handles and 'handle' in node['name'].lower():
            continue
        G.add_node(node['id'])
        if node['parent'] != -1:
            G.add_edge(node['id'], node['parent'])
    hashcode = nx.weisfeiler_lehman_graph_hash(G)
    return hashcode

# def save_cat_hashbook(root):
#     hashbook = {}
#     for cat in os.listdir(root):
#         hashbook[cat] = {}
#         for case in os.listdir(os.path.join(root, cat)):
#             file = json.load(open(os.path.join(root, cat, case, 'train_renumber.json'), 'r'))
#             code = get_hash(file, ignore_handles=False, dag=True)
#             file['meta']['tree_hash'] = code
#             # json.dump(file, open(os.path.join(root, cat, case, 'train_renumber.json'), 'w'))
#             if code not in hashbook[cat].keys():
#                 hashbook[cat][code] = [case]
#             else:
#                 hashbook[cat][code].append(case)
#     json.dump(hashbook, open('hashbook.json', 'w'))

# def save_across_cat_hashbook(root, out_name, mode):
#     postfix = 'renumber' if mode == 'our' else 'nap'
#     tree_key = 'diffuse_tree' if mode == 'our' else 'arti_tree'
#     hashbook = {}
#     for cat in os.listdir(root):
#         case_list = os.listdir(os.path.join(root, cat))
#         for case in case_list:
#             file = json.load(open(os.path.join(root, cat, case, f'train_{postfix}.json'), 'r'))
#             code = get_hash(file, tree_key, ignore_handles=False, dag=True)
#             file['meta']['tree_hash'] = code
#             info = f'{cat}/{case}'
#             # json.dump(file, open(os.path.join(root, cat, case, f'train_{postfix}.json'), 'w'))
#             if code not in hashbook.keys():
#                 hashbook[code] = [info]
#             else:
#                 hashbook[code].append(info)
#     json.dump(hashbook, open(f'hashbook{out_name}.json', 'w'))

    
            
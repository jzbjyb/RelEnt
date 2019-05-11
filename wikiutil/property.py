from typing import List, Tuple, Dict
from collections import defaultdict
import subprocess


def hiro_subgraph_to_tree_dict(root: str,
						  hiro_subg: List[Tuple[List[str], str, int]],
						  max_hop=1) -> Tuple[str, Dict[str, Tuple[str, Dict]]]:
	# group all the entities by their depth
	hop_dict: Dict[int, List[Tuple[List[str], str, int]]] = defaultdict(lambda: [])
	for e in hiro_subg:
		depth = e[2]
		hop_dict[depth].append(e)
	# generate subgraph dict from small hop to large hop
	# recursive tree structure that cannot handle cycles
	# TODO: hiro's graph is acyclic by its nature and might not be complete
	# TODO: hiro's graph is not complete comparing to the wikidata page and some properties are duplicate
	tree_dict: Tuple[Dict[str, Tuple[str, Dict]]] = (root, {})
	for hop in range(max_hop):
		hop += 1
		for e in hop_dict[hop]:
			plist, tid, _ = e
			trace = tree_dict[1]
			parent = None
			for p in plist[:-1]:
				parent = trace[p][0]
				trace = trace[p][1]
			trace[plist[-1]] = (tid, {})
	return tree_dict


def tree_dict_to_adj(tree_dict: Tuple[str, Dict[str, Tuple[str, Dict]]]) -> List[Tuple[str, str, str]]:
	# DFS to get all the connections used to construct adjacency matrix
	adjs: List[Tuple[str, str, str]] = []
	root = tree_dict[0]
	for p in tree_dict[1]:
		sub_tree_dict = tree_dict[1][p]
		adjs.append((root, p, sub_tree_dict[0]))
		adjs.extend(tree_dict_to_adj(sub_tree_dict))
	return adjs


def get_sub_properties(pid):
	output = subprocess.check_output(['wdtaxonomy', pid, '-f', 'csv'])
	output = output.decode('utf-8')
	subs = []
	for l in output.split('\n')[1:]:
		if l.startswith('-,'):  # first-level sub props
			l = l.split(',')
			subs.append((l[1], l[2].strip('"')))
	return subs


def read_subprop_file(filepath):
	result: List[Tuple[Tuple[str, str], List[Tuple[str, str]]]] = []
	with open(filepath, 'r') as fin:
		for l in fin:
			ps = l.strip().split('\t')
			par_id, par_label = ps[0].split(',', 1)
			childs = [tuple(p.split(',')) for p in ps[1:]]
			result.append(((par_id, par_label), childs))
	return result


def get_subtree(root: str, child_dict: Dict[str, List[str]]):
	if root not in child_dict:
		return (root, [])
	result = (root, [get_subtree(c, child_dict) for c in child_dict[root]])
	return result


def get_depth(root: Tuple[str, List]) -> int:
	depth = 1
	max_depth = 0
	for c in root[1]:
		d = get_depth(c)
		if d > max_depth:
			max_depth = d
	return depth + max_depth


def print_subtree(root: Tuple[str, List], id2label: Dict[str, str]=None, prefix='') -> str:
	l = prefix + root[0] + ': ' + id2label[root[0]] if id2label else ''
	ls = []
	for c in root[1]:
		ls.append(print_subtree(c, id2label, prefix + '\t'))
	return '\n'.join([l] + ls)

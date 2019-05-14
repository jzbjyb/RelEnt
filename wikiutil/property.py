from typing import List, Tuple, Dict
from collections import defaultdict
from itertools import combinations
import subprocess, re, os
import numpy as np


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


def read_prop_occ_file_from_dir(prop: str, dir: str, filter=False, use_order=False):
	filepath = os.path.join(dir, prop + '.txt')
	if os.path.exists(filepath):
		return read_prop_occ_file(filepath, filter=filter)
	if use_order:
		filepath = os.path.join(dir, prop + '.txt.order')
		if os.path.exists(filepath):
			return read_prop_occ_file(filepath, filter=filter)
	raise Exception('{} not exist'.format(prop))


def read_prop_occ_file(filepath, filter=False) -> List[Tuple[str, str]]:
	result = []
	with open(filepath, 'r') as fin:
		for l in fin:
			hid, _, tid, _ = l.strip().split('\t')
			if filter and not re.match('^Q[0-9]+$', hid) or not re.match('^Q[0-9]+$', tid):
				# only keep entities
				continue
			result.append((hid, tid))
	return result


def read_subprop_file(filepath) -> List[Tuple[Tuple[str, str], List[Tuple]]]:
	result: List[Tuple[Tuple[str, str], List[Tuple[str, str]]]] = []
	with open(filepath, 'r') as fin:
		for l in fin:
			ps = l.strip().split('\t')
			par_id, par_label = ps[0].split(',', 1)
			childs = [tuple(p.split(',')) for p in ps[1:]]
			result.append(((par_id, par_label), childs))
	return result


def read_pointiwse_file(filepath) \
		-> List[Tuple[Tuple[str, str, str], Tuple[str, str, str], int]]:
	result = []
	with open(filepath, 'r') as fin:
		for l in fin:
			label, p1o, p2o = l.strip().split('\t')
			label = int(label)
			p1, h1, t1 = p1o.split(' ')
			p2, h2, t2 = p2o.split(' ')
			result.append(((h1, p1, t1), (h2, p2, t2), label))
	return result


def read_subgraph_file(filepath) -> Dict[str, List[Tuple[str, str, str]]]:
	result = {}
	with open(filepath, 'r') as fin:
		for l in fin:
			l = l.strip().split('\t')
			root = l[0]
			adjs = [tuple(adj.split(' ')) for adj in l[1:]]
			result[root] = adjs
	return result


def get_subtree(root: str, child_dict: Dict[str, List[str]]) -> Tuple[str, List[Tuple[str, List]]]:
	if root not in child_dict:
		return (root, [])
	result = (root, [get_subtree(c, child_dict) for c in child_dict[root]])
	return result


def get_is_sibling(subprops: List[Tuple[Tuple[str, str], List[Tuple]]]):
	is_sibling = set()
	for p in subprops:
		for p1, p2 in combinations(p[1], 2):
			is_sibling.add((p1[0], p2[0]))
			is_sibling.add((p2[0], p1[0]))
	return is_sibling


def get_all_subtree(subprops: List[Tuple[Tuple[str, str], List[Tuple]]]) \
		-> Tuple[List[Tuple[str, List[Tuple]]], List[Tuple[str, List[Tuple]]]]:
	num_prop = len(subprops)
	print('{} props'.format(num_prop))

	# get parent link and children link
	parent_dict = defaultdict(lambda: [])
	child_dict = defaultdict(lambda: [])
	for p in subprops:
		parent_id = p[0][0]
		child_dict[parent_id] = [c[0] for c in p[1]]
		for c in p[1]:
			parent_dict[c[0]].append(parent_id)

	# construct tree for properties without parent
	subtrees: List[Tuple[str, List[Tuple]]] = []
	isolate: List[Tuple[str, List[Tuple]]] = []
	for p in subprops:
		pid = p[0][0]
		if len(parent_dict[pid]) == 0:
			subtree = get_subtree(pid, child_dict)
			if len(subtree[1]) > 0:
				subtrees.append(subtree)
			else:
				isolate.append(subtree)

	print('{} subtree'.format(len(subtrees)))
	print('avg depth: {}'.format(np.mean([get_depth(s) for s in subtrees])))
	print('{} isolated prop'.format(len(isolate)))

	return subtrees, isolate


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


class PropertySubtree():  # TODO: replace tuple with PropertySubtree
	def __init__(self, tree: Tuple[str, List]):
		self.tree = tree


	@property
	def nodes(self):
		return list(self.self_traverse_subtree())


	@staticmethod
	def traverse_subtree(subtree):
		yield subtree[0]
		for c in subtree[1]:
			yield from PropertySubtree.traverse_subtree(c)


	def self_traverse_subtree(self):
		yield from PropertySubtree.traverse_subtree(self.tree)


	@staticmethod
	def split_within_subtree(subtree, tr, dev, te):
		''' split the subtree by spliting each tir into train/dev/test set '''
		siblings = [c[0] for c in subtree[1]]
		tr = int(len(siblings) * tr)
		dev = int(len(siblings) * dev)
		te = len(siblings) - tr - dev
		test = siblings[tr + dev:]
		dev = siblings[tr:tr + dev]
		train = siblings[:tr]
		if len(train) > 0 and len(dev) > 0 and len(test) > 0:
			yield train, dev, test
		for c in subtree[1]:
			PropertySubtree.split_within_subtree(c, tr, dev, te)


	def self_split_within_subtree(self, tr, dev, te):
		yield from PropertySubtree.split_within_subtree(self.tree, tr, dev, te)


	@staticmethod
	def get_depth(subtree: Tuple[str, List]) -> int:
		depth = 1
		max_depth = 0
		for c in subtree[1]:
			d = PropertySubtree.get_depth(c)
			if d > max_depth:
				max_depth = d
		return depth + max_depth


	@staticmethod
	def print_subtree(subtree: Tuple[str, List],
					  id2label: Dict[str, str] = None,
					  defalut_label: str = '',
					  prefix: str = '') -> str:
		id = subtree[0]
		label = id2label[id] if id2label and id in id2label else defalut_label
		l = prefix + id + ': ' + label
		ls = []
		for c in subtree[1]:
			ls.append(PropertySubtree.print_subtree(
				c, id2label=id2label, defalut_label=defalut_label, prefix=prefix + '\t'))
		return '\n'.join([l] + ls)


	def self_print_subtree(self,
						   id2label: Dict[str, str] = None,
						   defalut_label: str = '',
						   prefix: str = '') -> str:
		return PropertySubtree.print_subtree(
			self.tree, id2label=id2label, defalut_label=defalut_label, prefix=prefix)

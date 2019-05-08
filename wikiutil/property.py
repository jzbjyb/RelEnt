from typing import List, Tuple, Dict
import subprocess


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

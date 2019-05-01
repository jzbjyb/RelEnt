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

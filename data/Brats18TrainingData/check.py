import os
import shutil

with open('output.txt','r') as f:
	lines = f.readlines()

for i, o in enumerate(lines):
	lines[i] = lines[i].strip()
        folder = lines[i]
	lines_old = lines[i]+ '/'+ lines[i][6:] + '_t1.nii.gz'
	lines_new = lines[i]+ '/'+ lines[i][6:] + '_seg.nii.gz'
        shutil.copy(lines_old, lines_new)
	print(lines[i])


import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='', help='The checkpoint to be filtered')
parser.add_argument('--blacklist', type=str, default='fastrcnn/outputs/,maskrcnn/conv/,cascade_rcnn_stage1/outputs/,cascade_rcnn_stage2/outputs/,cascade_rcnn_stage3/outputs/', help='Blacklist nodes')

args = parser.parse_args()

if not args.ckpt or os.path.splitext(args.ckpt)[-1] != '.npz' or not os.path.isfile(args.ckpt):
	raise Exception('==> Invalid checkpoint')

weights = np.load(args.ckpt)
blacklist = args.blacklist.split(',')

keys = weights.keys()
weights = {key: weights[key] for key in keys}
for key in keys:
	for x in blacklist:
		if key.startswith(x):
			weights.pop(key)

out_path = os.path.splitext(args.ckpt)[0] + '_filtered.npz'
np.savez(out_path, **weights)

import os
import random
import itertools
import colorsys
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon
import IPython.display
import utils


def display_images(images, titles=None, cols=4, cmap=None, norm=None, interpolation=None):
	titles = titles if titles is not None else [""] * len(images)
	rows = len(images) // cols + 1
	plt.figure(figsize=(14, 14 * rows // cols))
	i = 1
	for image, title in zip(images, titles):
		plt.subplot(rows, cols, i)
		plt.title(title, fontsize=9)
		plt.axis("off")
		plt.imshow(image.astype(np.uint8), cmap=cmap, norm=norm, interpolation=interpolation)
		i += 1
	plt.show()

def random_colors(N, bright=True):
	brightness = 1.0 if bright else 0.7
	hsv = [(i / N, 1, brightness) for i in range(N)]
	colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	random.shuffle(colors)
	return colors

def apply_mask(image, mask, color, alpha=0.5):
	for c in range(3):
		image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + color[c] * 255 * alpha, image[:, :, c])
	return image

def display_instances(image, boxes, masks, class_ids, class_names, scores=None, title="", figsize=(10, 10), ax=None, 
	show_mask=True, show_bbox=True, colors=None, captions=None):
	N = boxes.shape[0]
	if not N:
		print("\n*** No instances to display *** \n")
	else:
		assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

	auto_show = False
	if not ax:
		_, ax = plt.subplots(1, figsize=figsize)
		auto_show = True

	colors = colors or random_colors(N)

	height, width = image.shape[0], image.shape[1]
	ax.set_ylim(height + 10, -10)
	ax.set_xlim(-10, width + 10)
	ax.axis("off")
	ax.set_title(title)

	masked_image = image.astype(np.uint32).copy()

	for i in range(N):
		color = colors[i]
		if not np.any(boxes[i]):
			continue
		y1, x1, y2, x2 = boxes[i]
		if show_bbox:
			p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, alpha=0.7, linestyle="dashed", edgecolor=color, 
				facecolor="none")
			ax.add_patch(p)
		if not captions:
			class_id = class_ids[i]
			score = scores[i] if scores is not None else None
			label = class_names[class_id]
			caption = "{} {:.3f}".format(label, score) if score else label
		else:
			caption = captions[i]
		ax.text(x1, y1, caption, color="w", size=11, backgroundcolor="none")
		mask = masks[:, :, i]
		if show_mask:
			masked_image = apply_mask(masked_image, mask, color)
		padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
		padded_mask[1:-1, 1:-1] = mask
		contours = find_contours(padded_mask, 0.5)
		for verts in contours:
			verts = np.fliplr(verts) - 1
			p = Polygon(verts, facecolor="none", edgecolor=color)
			ax.add_patch(p)
	ax.imshow(masked_image.astype(np.uint8))
	if auto_show:
		plt.show()

def display_differences(image, gt_box, gt_class_id, gt_mask, pred_box, pred_class_id, pred_score, pred_mask, class_names, 
	title="", ax=None, show_mask=True, show_box=True, iou_threshold=0.5, score_threshold=0.5):
	gt_match, pred_match, overlaps = utils.compute_matches(gt_box, gt_class_id, gt_mask, pred_box, pred_class_id, pred_score, 
		pred_mask, iou_threshold=iou_threshold, score_threshold=score_threshold)
	colors = [(0, 1, 0, 0.8)] * len(gt_match) + [(1, 0, 0, 1)] * len(pred_match)

	class_ids = np.concatenate([gt_class_id, pred_class_id])
	scores = np.concatenate([np.zeros((len(gt_match))), pred_score])
	boxes = np.concatenate([gt_box, pred_box])
	masks = np.concatenate([gt_mask, pred_mask], axis=-1)

	captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(pred_score[i], (overlaps[i, int(pred_match[i])] if 
		pred_match[i] > -1 else overlaps[i].max())) for i in range(len(pred_match))]
	title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
	display_instances(image, boxes, masks, class_ids, class_names, scores, title=title, ax=ax, show_mask=show_mask, 
		show_bbox=show_bbox, colors=colors, captions=captions)
import tensorflow as tf
import numpy as np
import random
import skimage.color
import skimage.io
import skimage.transform
import scipy
import logging


def generate_anchors(scale, ratios, shape, feature_stride):
	scales, ratios = np.meshgrid(np.array(scale), np.array(ratios))
	scales, ratios = scales.flatten(), ratios.flatten()
	heights, widths = scales / np.sqrt(ratios), scales * np.sqrt(ratios)

	shift_y = np.arange(0, shape[0], 1) * feature_stride
	shift_x = np.arange(0, shape[1], 1) * feature_stride
	shift_x, shift_y = np.meshgrid(shift_x, shift_y)

	box_widths, box_centers_x = np.meshgrid(widths, shift_x)
	box_heights, box_centers_y = np.meshgrid(heights, shift_y)

	box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
	box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

	boxes = np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)

	return boxes

def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides):
	anchors = []
	for i in range(len(scales)):
		anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i], feature_strides[i]))
	return np.concatenate(anchors, axis=0)

def norm_boxes(boxes, shape):
	h, w = shape
	scale = np.array([h - 1, w - 1, h - 1, w - 1])
	shift = np.array([0, 0, 1, 1])
	return np.divide((boxes - shift), scale).astype(np.float32)

def denorm_boxes(boxes, shape):
	h, w = shape
	scale = np.array([h - 1, w - 1, h - 1, w - 1])
	shift = np.array([0, 0, 1, 1])
	return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)

def batch_slic(inputs, graph_fn, batch_size, name=None):
	if not isinstance(inputs, list):
		inputs = [inputs]

	outputs = []
	for i in range(batch_size):
		input_slice = [x[i] for x in inputs]
		output_slice = graph_fn(*input_slice)
		if not isinstance(output_slice, (list, tuple)):
			output_slice = [output_slice]
		outputs.append(output_slice)
	outputs = list(zip(*outputs))

	if name is None:
		name = [None] * len(outputs)

	result = [tf.stack(o, axis=0, name=n) for o, n in zip(outputs, name)]

	if len(result) == 1:
		result = result[0]

	return result

def box_refinement_graph(box, gt_box):
	box = tf.cast(box, tf.float32)
	gt_box = tf.cast(gt_box, tf.float32)

	height = box[:, 2] - box[:, 0]
	width = box[:, 3] - box[:, 1]
	center_y = box[:, 0] + 0.5 * height
	center_x = box[:, 1] + 0.5 * width

	gt_height = gt_box[:, 2] - gt_box[:, 0]
	gt_width = gt_box[:, 3] - gt_box[:, 1]
	gt_center_y = gt_box[:, 0] + 0.5 * gt_height
	gt_center_x = gt_box[:, 1] + 0.5 * gt_width

	dy = (gt_center_y - center_y) / height
	dx = (gt_center_x - center_x) / width
	dh = tf.math.log(gt_height / height)
	dw = tf.math.log(gt_width / width)

	result = tf.stack([dy, dx, dh, dw], axis=1)

	return result

def resize(image, output_shape, order=1, mode="constant", cval=0, clip=True, preserve_range=False, 
	anti_aliasing=False, anti_aliasing_sigma=None):
	return skimage.transform.resize(image, output_shape, order=order, mode=mode, cval=cval, clip=clip, 
		preserve_range=preserve_range, anti_aliasing=anti_aliasing, anti_aliasing_sigma=anti_aliasing_sigma)

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
	image_dtype = image.dtype
	h, w = image.shape[:2]
	window = [0, 0, h, w]
	scale = 1
	padding = [(0, 0), (0, 0), (0, 0)]
	crop = None

	if mode is None:
		return image, window, scale, padding, crop

	if min_dim:
		scale = max(1., min_dim / min(h, w))
	if min_scale and min_scale > scale:
		scale = min_scale

	if max_dim and mode == "square":
		image_max = max(h, w)
		if round(image_max * scale > max_dim):
			scale = max_dim / image_max

	if scale != 1:
		image = resize(image, (round(h * scale), round(w * scale)), preserve_range=True)

	if mode == "square":
		h, w = image.shape[:2]
		top_pad = (max_dim - h) // 2
		bottom_pad = max_dim - top_pad - h
		left_pad = (max_dim - w) // 2
		right_pad = max_dim - left_pad - w
		padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
		image = np.pad(image, padding, mode="constant", constant_values=0)
		window = [top_pad, left_pad, top_pad + h, left_pad + w]
	elif mode == "pad64":
		h, w = image.shape[:2]
		assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
		if h % 64 > 0:
			max_h = h - (h % 64) + 64
			top_pad = (max_h - h) // 2
			bottom_pad = max_h - top_pad - h
		else:
			top_pad = bottom_pad = 0
		if w % 64 > 0:
			max_w = w - (w % 64) + 64
			left_pad = (max_w - w) // 2
			right_pad = max_w - left_pad - w
		else:
			left_pad = right_pad = 0
		padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
		image = np.pad(image, padding, mode="constant", constant_values=0)
		window = [top_pad, left_pad, top_pad + h, left_pad + w]
	elif mode == "crop":
		h, w = image.shape[:2]
		y = random.randint(0, (h - min_dim))
		w = random.randint(0, (w - min_dim))
		crop = [y, x, y + min_dim, x + min_dim]
		image = image[y : y + min_dim, x : x + min_dim]
		window = [0, 0, min_dim, min_dim]
	else:
		raise Exception("Mode {} not supported".format(mode))
	return image.astype(image_dtype), window, scale, padding, crop

def resize_mask(mask, scale, padding, crop=None):
	mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
	if crop is not None:
		y, x, h, w = crop
		mask = mask[y : y + h, x : x + w]
	else:
		mask = np.pad(mask, padding, mode="constant", constant_values=0)
	return mask

def unmold_mask(mask, bbox, image_shape, threshold=0.5):
	y1, x1, y2, x2 = bbox
	mask = resize(mask, (y2 - y1, x2 - x1))
	mask = np.where(mask >= threshold, 1, 0).astype(np.bool)
	full_mask = np.zeros(image_shape[:2], dtype=np.bool)
	full_mask[y1:y2, x1:x2] = mask
	return full_mask

def extract_bboxes(mask):
	boxes = np.zeros((mask.shape[-1], 4), dtype=np.int32)
	for i in range(mask.shape[-1]):
		m = mask[:, :, i]
		horizontal_indices = np.where(np.any(m, axis=0))[0]
		vertical_indices = np.where(np.any(m, axis=1))[0]
		if horizontal_indices.shape[0] and vertical_indices.shape[0]:
			x1, x2 = horizontal_indices[[0, -1]]
			y1, y2 = vertical_indices[[0, -1]]
			x2 += 1
			y2 += 1
		else:
			x1, y1, x2, y2 = 0, 0, 0, 0
		boxes[i] = np.array([y1, x1, y2, x2])
	return boxes.astype(np.int32)

def minimize_mask(bbox, mask, mini_shape):
	mini_mask = np.zeros(mini_shape + [mask.shape[-1],], dtype=bool)
	for i in range(mask.shape[-1]):
		m = mask[:, :, i].astype(bool)
		y1, x1, y2, x2 = bbox[i][:4]
		m = m[y1:y2, x1:x2]
		if m.size == 0:
			raise Exception("Invalid bounding box with area of zero")
		m = resize(m, mini_shape)
		mini_mask[:, :, i] = np.around(m).astype(np.bool)
	return mini_mask

def compute_iou(box, boxes, box_area, boxes_area):
	y1 = np.maximum(box[0], boxes[:, 0])
	x1 = np.maximum(box[1], boxes[:, 1])
	y2 = np.minimum(box[2], boxes[:, 2])
	x2 = np.minimum(box[3], boxes[:, 3])

	intersection = np.maximum(y2 - y1, 0) * np.maximum(x2 - x1, 0)
	union = box_area + boxes_area[:] - intersection[:]
	iou = intersection / union

	return iou

def compute_overlaps(boxes1, boxes2):
	area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
	area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
	overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
	for i in range(overlaps.shape[1]):
		box2 = boxes2[i]
		overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
	return overlaps

def compute_iou(box, boxes, box_area, boxes_area):
	y1 = np.maximum(box[0], boxes[:, 0])
	x1 = np.maximum(box[1], boxes[:, 1])
	y2 = np.minimum(box[2], boxes[:, 2])
	x2 = np.minimum(box[3], boxes[:, 3])
	intersection = np.maximum(y2 - y1, 0) * np.maximum(x2 - x1, 0)
	union = box_area + boxes_area[:] - intersection[:]
	iou = intersection / union
	return iou

def box_refinement(box, gt_box):
	box = box.astype(np.float32)
	gt_box = gt_box.astype(np.float32)

	height = box[:, 2] - box[:, 0]
	width = box[:, 3] - box[:, 1]
	center_y = box[:, 0] + 0.5 * height
	center_x = box[:, 1] + 0.5 * width

	gt_height = gt_box[:, 2] - gt_box[:, 0]
	gt_width = gt_box[:, 3] - gt_box[:, 1]
	gt_center_y = gt_box[:, 0] + 0.5 * gt_height
	gt_center_x = gt_box[:, 1] + 0.5 * gt_width

	dy = (gt_center_y - center_y) / height
	dx = (gt_center_x - center_x) / width
	dh = np.log(gt_height / height)
	dw = np.log(gt_width / width)

	return np.stack([dy, dx, dh, dw], axis=1)

def trim_zeros(x):
	assert len(x.shape) == 2
	return x[~np.all(x == 0, axis=1)]

def compute_overlaps_masks(masks1, masks2):
	if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
		return np.zeros((masks1.shape[-1], masks2.shape[-1]))

	masks1 = np.reshape(masks1 > 0.5, (-1, masks1.shape[-1])).astype(np.float32)
	masks2 = np.reshape(masks2 > 0.5, (-1, masks2.shape[-1])).astype(np.float32)
	area1 = np.sum(masks1, axis=0)
	area2 = np.sum(masks2, axis=0)

	intersection = np.dot(masks1.T, masks2)
	union = area1[:, None] + area2[None, :] - intersection
	overlaps = intersection / union

	return overlaps

def compute_matches(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks, iou_threshold=0.5,
	score_threshold=0.0):
	gt_boxes = trim_zeros(gt_boxes)
	gt_masks = gt_masks[..., :gt_boxes.shape[0]]
	pred_boxes = trim_zeros(pred_boxes)
	pred_scores = pred_scores[:pred_boxes.shape[0]]

	indices = np.argsort(pred_scores)[::-1]
	pred_boxes = pred_boxes[indices]
	pred_class_ids = pred_class_ids[indices]
	pred_scores = pred_scores[indices]
	pred_masks = pred_masks[..., indices]

	overlaps = compute_overlaps_masks(pred_masks, gt_masks)

	match_count = 0
	pred_match = -1 * np.ones((pred_boxes.shape[0]))
	gt_match = -1 * np.ones((gt_boxes.shape[0]))
	for i in range(len(pred_boxes)):
		sorted_ixs = np.argsort(overlaps[i])[::-1]
		low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
		if low_score_idx.size > 0:
			sorted_ixs = sorted_ixs[:low_score_idx[0]]
		for j in sorted_ixs:
			if gt_match[j] > -1:
				continue
			iou = overlaps[i, j]
			if iou < iou_threshold:
				break
			if pred_class_ids[i] == gt_class_ids[j]:
				match_count += 1
				pred_match[i] = j
				gt_match[j] = i
				break

	return gt_match, pred_match, overlaps


class Dataset():
	def __init__(self, class_map=None):
		self._image_ids = []
		self.image_info = []
		self.class_info = [{"source": "", "id": 0, "name": "BG"}]
		self.source_class_ids = {}

	def add_class(self, source, class_id, class_name):
		assert "." not in source, "Source name cannot contain a dot"
		for info in self.class_info:
			if info["source"] == source and info["id"] == class_id:
				return
		self.class_info.append({"source": source, "id": class_id, "name": class_name})

	def add_image(self, source, image_id, path, **kwargs):
		image_info = {"id": image_id, "source": source, "path": path}
		image_info.update(kwargs)
		self.image_info.append(image_info)

	def image_reference(self, image_id):
		return ""

	def prepare(self, class_map=None):
		def clean_name(name):
			return ",".join(name.split(",")[:1])

		self.num_classes = len(self.class_info)
		self.class_ids = np.arange(self.num_classes)
		self.class_names = [clean_name(c["name"]) for c in self.class_info]
		self.num_images = len(self.image_info)
		self._image_ids = np.arange(self.num_images)
		self.class_from_source_map = {"{}.{}".format(info["source"], info["id"]): id for info, id in zip(self.class_info, 
			self.class_ids)}
		self.image_from_source_map = {"{}.{}".format(info["source"], info["id"]): id for info, id in zip(self.image_info, 
			self.image_ids)}
		self.sources = list(set([i["source"] for i in self.class_info]))
		self.source_class_ids = {}

		for source in self.sources:
			self.source_class_ids[source] = []
			for i, info in enumerate(self.class_info):
				if i == 0 or source == info["source"]:
					self.source_class_ids[source].append(i)

	def map_source_class_id(self, source_class_id):
		return self.class_from_source_map[source_class_id]

	def get_source_class_id(self, class_id, source):
		info = self.class_info[class_id]
		assert info["source"] == source
		return info["id"]

	@property
	def image_ids(self):
		return self._image_ids
	
	def source_image_linek(self, image_id):
		return self.image_info[image_id]["path"]

	def load_image(self, image_id):
		image = skimage.io.imread(self.image_info[image_id]["path"])
		if image.ndim != 3:
			image = skimage.color.gray2rgb(image)
		if image.shape[-1] == 4:
			image = image[:, :, :3]
		return image

	def load_mask(self, image_id):
		logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
		mask = np.empty([0, 0, 0])
		class_ids = np.empty([0], np.int32)
		return mask, class_ids
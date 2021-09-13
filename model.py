import tensorflow as tf
import numpy as np
import cv2
import os
import re
import random
import datetime
import math
import logging
import h5py
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import utils
from keras.engine import topology as saving
from config import Config


def log(text, array=None):
	if array is not None:
		text = text.ljust(25)
		text += ("shape: {:20}  ".format(str(array.shape)))
		if array.size:
			text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(), array.max()))
		else:
			text += ("min: {:10}  max: {:10}".format("", ""))
		text += "  {}".format(array.dtype)
	print(text)

def norm_boxes_graph(boxes, shape):
	h, w = tf.split(tf.cast(shape, tf.float32), 2)
	scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
	shift = tf.constant([0.0, 0.0, 1.0, 1.0])
	return tf.math.divide(boxes - shift, scale)

def conv_block(inputs, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
	filters_1, filters_2, filters_3 = filters
	conv_name_base = "res" + str(stage) + block + "_branch"
	bn_name_base = "bn" + str(stage) + block + "_branch"

	x = KL.Conv2D(filters_1, (1, 1), strides=strides, name=conv_name_base + "2a", use_bias=use_bias)(inputs)
	x = BatchNorm(name=bn_name_base + "2a")(x, training=train_bn)
	x = KL.Activation("relu")(x)

	x = KL.Conv2D(filters_2, (kernel_size, kernel_size), padding="SAME", name=conv_name_base + "2b", use_bias=use_bias)(x)
	x = BatchNorm(name=bn_name_base + "2b")(x, training=train_bn)
	x = KL.Activation("relu")(x)

	x = KL.Conv2D(filters_3, (1, 1), name=conv_name_base + "2c", use_bias=use_bias)(x)
	x = BatchNorm(name=bn_name_base + "2c")(x, training=train_bn)

	shortcut = KL.Conv2D(filters_3, (1, 1), strides=strides, name=conv_name_base + "1", use_bias=use_bias)(inputs)
	shortcut = BatchNorm(name=bn_name_base + "1")(shortcut, training=train_bn)

	x = KL.Add()([x, shortcut])
	x = KL.Activation("relu", name="res" + str(stage) + block + "_out")(x)

	return x

def identity_block(inputs, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
	filters_1, filters_2, filters_3 = filters
	conv_name_base = "res" + str(stage) + block + "_branch"
	bn_name_base = "bn" + str(stage) + block + "_branch"

	x = KL.Conv2D(filters_1, (1, 1), name=conv_name_base + "2a", use_bias=use_bias)(inputs)
	x = BatchNorm(name=bn_name_base + "2a")(x, training=train_bn)
	x = KL.Activation("relu")(x)

	x = KL.Conv2D(filters_2, (kernel_size, kernel_size), padding="SAME", name=conv_name_base + "2b", use_bias=use_bias)(x)
	x = BatchNorm(name=bn_name_base + "2b")(x, training=train_bn)
	x = KL.Activation("relu")(x)

	x = KL.Conv2D(filters_3, (1, 1), name=conv_name_base + "2c", use_bias=use_bias)(x)
	x = BatchNorm(name=bn_name_base + "2c")(x, training=train_bn)

	x = KL.Add()([x, inputs])
	x = KL.Activation("relu", name="res" + str(stage) + block + "_out")(x)

	return x

def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
	assert architecture in ["resnet50", "resnet101"]

	x = KL.ZeroPadding2D((3, 3))(input_image)
	x = KL.Conv2D(64, (7, 7), strides=(2, 2), name="conv1", use_bias=True)(x)
	x = BatchNorm(name="bn_conv1")(x, training=train_bn)
	x = KL.Activation("relu")(x)
	C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="SAME")(x)

	x = conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1), train_bn=train_bn)
	x = identity_block(x, 3, [64, 64, 256], stage=2, block="b", train_bn=train_bn)
	C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block="c", train_bn=train_bn)

	x = conv_block(x, 3, [128, 128, 512], stage=3, block="a", train_bn=train_bn)
	x = identity_block(x, 3, [128, 128, 512], stage=3, block="b", train_bn=train_bn)
	x = identity_block(x, 3, [128, 128, 512], stage=3, block="c", train_bn=train_bn)
	C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block="d", train_bn=train_bn)

	x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a", train_bn=train_bn)
	block_cout = {"resnet50": 5, "resnet101": 22}[architecture]
	for i in range(block_cout):
		x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
	C4 = x

	if stage5:
		x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a", train_bn=train_bn)
		x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b", train_bn=train_bn)
		C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c", train_bn=train_bn)
	else:
		C5 = None

	return [C1, C2, C3, C4, C5]

def build_rpn_model(anchors_per_location, depth):
	input_feature_map = KL.Input(shape=[None, None, depth], name="input_rpn_feature_map")
	shared = KL.Conv2D(512, (3, 3), padding="SAME", activation="relu", name="rpn_conv_shared")(input_feature_map)

	x = KL.Conv2D(anchors_per_location * 2, (1, 1), padding="VALID", activation="linear", name="rpn_class_raw")(shared)
	rpn_class_logits = KL.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 2]))(x)
	rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

	x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="VALID", activation="linear", name="rpn_bbox_pred")(shared)
	rpn_bbox = KL.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 4]))(x)

	return KM.Model([input_feature_map], [rpn_class_logits, rpn_probs, rpn_bbox], name="rpn_model")

def apply_box_deltas_graph(boxes, deltas):
	height = boxes[:, 2] - boxes[:, 0]
	width = boxes[:, 3] - boxes[:, 1]
	center_y = boxes[:, 0] + 0.5 * height
	center_x = boxes[:, 1] + 0.5 * width

	center_y += deltas[:, 0] * height
	center_x += deltas[:, 1] * width
	height *= tf.exp(deltas[:, 2])
	width *= tf.exp(deltas[:, 3])

	y1 = center_y - 0.5 * height
	x1 = center_x - 0.5 * width
	y2 = y1 + height
	x2 = x1 + width
	result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")

	return result

def clip_boxes_graph(boxes, window):
	wy1, wx1, wy2, wx2 = tf.split(window, 4)
	y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)

	y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
	x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
	y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
	x2 = tf.maximum(tf.minimum(x2, wx2), wx1)

	clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
	clipped.set_shape((clipped.shape[0], 4))

	return clipped

def parse_image_meta_graph(meta):
	image_id = meta[:, 0]
	original_image_shape = meta[:, 1:4]
	image_shape = meta[:, 4:7]
	window = meta[:, 7:11]
	scale = meta[:, 11]
	active_class_ids = meta[:, 12:]
	return {"image_id": image_id, "original_image_shape": original_image_shape,
	"image_shape": image_shape, "window": window, "scale": scale, "active_class_ids": active_class_ids}

def trim_zeros_graph(boxes, name="trim_zeros"):
	non_zeros = tf.cast(tf.math.reduce_sum(tf.math.abs(boxes), axis=1), tf.bool)
	boxes = tf.boolean_mask(boxes, non_zeros, name=name)
	return boxes, non_zeros

def overlaps_graph(boxes1, boxes2):
	b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
	b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
	b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
	b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
	y1 = tf.math.maximum(b1_y1, b2_y1)
	x1 = tf.math.maximum(b1_x1, b2_x1)
	y2 = tf.math.minimum(b1_y2, b2_y2)
	x2 = tf.math.minimum(b1_x2, b2_x2)
	intersection = tf.math.maximum(y2 - y1, 0) * tf.math.maximum(x2 - x1, 0)
	b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
	b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
	union = b1_area + b2_area - intersection
	iou = intersection / union
	overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
	return overlaps

def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
	asserts = [tf.Assert(tf.math.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion")]
	with tf.control_dependencies(asserts):
		proposals = tf.identity(proposals)

	proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
	gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
	gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
	gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2, name="trim_gt_masks")

	crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
	non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
	crowd_boxes = tf.gather(gt_boxes, crowd_ix)
	gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
	gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
	gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

	overlaps = overlaps_graph(proposals, gt_boxes)
	crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
	crowd_iou_max = tf.math.reduce_max(crowd_overlaps, axis=1)
	no_crowd_bool = (crowd_iou_max < 0.001)

	roi_iou_max = tf.math.reduce_max(overlaps, axis=1)
	positive_roi_bool = (roi_iou_max >= 0.5)
	positive_indices = tf.where(positive_roi_bool)[:, 0]
	negative_indices = tf.where(tf.math.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

	positive_count = int(config.train_rois_per_image * config.roi_positive_ratio)
	positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
	positive_count = tf.shape(positive_indices)[0]

	r = 1.0 / config.roi_positive_ratio
	negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
	negative_indices = tf.random_shuffle(negative_indices)[:negative_count]

	positive_rois = tf.gather(proposals, positive_indices)
	negative_rois = tf.gather(proposals, negative_indices)

	positive_overlaps = tf.gather(overlaps, positive_indices)
	roi_gt_box_assignment = tf.cond(tf.math.greater(tf.shape(positive_overlaps)[1], 0), 
		true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
		false_fn = lambda: tf.cast(tf.constant([]), tf.int64))
	roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
	roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

	deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes) / config.bbox_std_dev

	transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
	roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

	boxes = positive_rois
	if config.use_mini_mask:
		y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
		gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
		gt_h, gt_w = gt_y2 - gt_y1, gt_x2 - gt_x1
		y1 = (y1 - gt_y1) / gt_h
		x1 = (x1 - gt_x1) / gt_w
		y2 = (y2 - gt_y1) / gt_h
		x2 = (x2 - gt_x1) / gt_w
		boxes = tf.concat([y1, x1, y2, x2], axis=1)
	box_ids = tf.range(0, tf.shape(roi_masks)[0])
	masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes, box_ids, config.mask_shape)
	masks = tf.round(tf.squeeze(masks, axis=3))

	rois = tf.concat([positive_rois, negative_rois], axis=0)
	N = tf.shape(negative_rois)[0]
	P = tf.math.maximum(config.train_rois_per_image - tf.shape(rois)[0], 0)
	rois = tf.pad(rois, [(0, P), (0, 0)])
	roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
	roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
	deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
	masks = tf.pad(masks, [(0, N + P), (0, 0), (0, 0)])

	return rois, roi_gt_class_ids, deltas, masks

def log2_graph(x):
	return tf.math.log(x) / tf.math.log(2.0)

def fpn_classifier_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True, fc_layers_size=1024):
	x = PyramidROIAlign([pool_size, pool_size], name="roi_align_classifier")([rois, image_meta] + feature_maps)
	x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="VALID"), name="mrcnn_class_conv1")(x)
	x = KL.TimeDistributed(BatchNorm(), name="mrcnn_class_bn1")(x, training=train_bn)
	x = KL.Activation("relu")(x)
	x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1), padding="VALID"), name="mrcnn_class_conv2")(x)
	x = KL.TimeDistributed(BatchNorm(), name="mrcnn_class_bn2")(x, training=train_bn)
	x = KL.Activation("relu")(x)
	shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)

	mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name="mrcnn_class_logits")(shared)
	mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)

	x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation="linear"), name="mrcnn_bbox_fc")(shared)
	s = K.int_shape(x)
	mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

	return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

def build_fpn_mask_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True):
	x = PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")([rois, image_meta] + feature_maps)
	x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="SAME"), name="mrcnn_mask_conv1")(x)
	x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn1")(x, training=train_bn)
	x = KL.Activation("relu")(x)
	x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="SAME"), name="mrcnn_mask_conv2")(x)
	x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn2")(x, training=train_bn)
	x = KL.Activation("relu")(x)
	x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="SAME"), name="mrcnn_mask_conv3")(x)
	x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn3")(x, training=train_bn)
	x = KL.Activation("relu")(x)
	x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="SAME"), name="mrcnn_mask_conv4")(x)
	x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn4")(x, training=train_bn)
	x = KL.Activation("relu")(x)
	x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv")(x)
	x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")(x)
	return x

def batch_pack_graph(x, counts, num_rois):
	outputs = []
	for i in range(num_rois):
		outputs.append(x[i, :counts[i]])
	return tf.concat(outputs, axis=0)

def smooth_l1_loss(y_true, y_pred):
	diff = K.abs(y_true - y_pred)
	less_than_one = K.cast(K.less(diff, 1.0), "float32")
	loss = (0.5 * less_than_one * diff**2) + (1.0 - less_than_one) * (diff - 0.5)
	return loss

def rpn_class_loss_graph(rpn_match, rpn_class_logits):
	rpn_match = tf.squeeze(rpn_match, axis=-1)
	anchor_class = tf.cast(K.equal(rpn_match, 1), tf.int32)
	indices = tf.where(K.not_equal(rpn_match, 0))
	rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
	anchor_class = tf.gather_nd(anchor_class, indices)
	loss = K.sparse_categorical_crossentropy(target=anchor_class, output=rpn_class_logits, from_logits=True)
	loss = K.switch(tf.size(loss)>0, K.mean(loss), tf.constant(0.0))
	return loss

def rpn_bbox_loss_graph(batch_size, target_bbox, rpn_match, rpn_bbox):
	rpn_match = K.squeeze(rpn_match, -1)
	indices = tf.where(K.equal(rpn_match, 1))
	rpn_bbox = tf.gather_nd(rpn_bbox, indices)
	batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
	target_bbox = batch_pack_graph(target_bbox, batch_counts, batch_size)
	loss = smooth_l1_loss(target_bbox, rpn_bbox)
	loss = K.switch(tf.size(loss)>0, K.mean(loss), tf.constant(0.0))
	return loss

def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
	target_class_ids = tf.cast(target_class_ids, tf.int64)
	pred_class_ids = tf.argmax(pred_class_logits, axis=2)
	pred_active = tf.gather(active_class_ids[0], pred_class_ids)
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=pred_class_logits)
	loss = loss * pred_active
	loss = tf.math.reduce_sum(loss) / tf.math.reduce_sum(pred_active)
	return loss

def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
	target_class_ids = K.reshape(target_class_ids, (-1,))
	target_bbox = K.reshape(target_bbox, (-1, 4))
	pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

	positive_roi_ix = tf.where(target_class_ids>0)[:, 0]
	positive_roi_class_idx = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
	indices = tf.stack([positive_roi_ix, positive_roi_class_idx], axis=1)

	target_bbox = tf.gather(target_bbox, positive_roi_ix)
	pred_bbox = tf.gather_nd(pred_bbox, indices)

	loss = K.switch(tf.size(target_bbox)>0, smooth_l1_loss(target_bbox, pred_bbox), tf.constant(0.0))
	loss = K.mean(loss)

	return loss

def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
	target_class_ids = K.reshape(target_class_ids, (-1,))
	mask_shape = tf.shape(target_masks)
	target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
	pred_shape = tf.shape(pred_masks)
	pred_masks = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
	pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])
	positive_ix = tf.where(target_class_ids>0)[:, 0]
	positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
	indices = tf.stack([positive_ix, positive_class_ids], axis=1)
	y_true = tf.gather(target_masks, positive_ix)
	y_pred = tf.gather_nd(pred_masks, indices)
	loss = K.switch(tf.size(y_true>0), K.binary_crossentropy(target=y_true, output=y_pred), tf.constant(0.0))
	loss = K.mean(loss)
	return loss

def refine_detections_graph(rois, probs, deltas, window, config):
	class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
	indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
	class_scores = tf.gather_nd(probs, indices)
	deltas_specific = tf.gather_nd(deltas, indices)
	refined_rois = apply_box_deltas_graph(rois, deltas_specific * config.bbox_std_dev)
	refined_rois = clip_boxes_graph(refined_rois, window)

	keep = tf.where(class_ids > 0)[:, 0]
	conf_keep = tf.where(class_scores >= config.detection_min_confidence)[:, 0]
	keep = tf.sets.intersection(tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
	keep = tf.sparse.to_dense(keep)[0]

	pre_nms_class_ids = tf.gather(class_ids, keep)
	pre_nms_scores = tf.gather(class_scores, keep)
	pre_nms_rois = tf.gather(refined_rois, keep)
	unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

	def nms_keep_map(class_id):
		ixs = tf.where(tf.math.equal(pre_nms_class_ids, class_id))[:, 0]
		class_keep = tf.image.non_max_suppression(tf.gather(pre_nms_rois, ixs), tf.gather(pre_nms_scores, ixs), 
			max_output_size=config.detection_max_instances, iou_threshold=config.detection_nms_threshold)
		class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
		gap = config.detection_max_instances - tf.shape(class_keep)[0]
		class_keep = tf.pad(class_keep, [(0, gap)], mode="CONSTANT", constant_values=-1)
		class_keep.set_shape([config.detection_max_instances])
		return class_keep

	nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)
	nms_keep = tf.reshape(nms_keep, [-1])
	nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
	keep = tf.sets.intersection(tf.expand_dims(keep, 0), tf.expand_dims(nms_keep, 0))
	keep = tf.sparse.to_dense(keep)[0]

	roi_count = config.detection_max_instances
	class_scores_keep = tf.gather(class_scores, keep)
	num_keep = tf.math.minimum(tf.shape(class_scores_keep)[0], roi_count)
	top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
	keep = tf.gather(keep, top_ids)

	detections = tf.concat([tf.gather(refined_rois, keep), tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis], 
		tf.gather(class_scores, keep)[..., tf.newaxis]], axis=1)
	gap = config.detection_max_instances - tf.shape(detections)[0]
	detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")

	return detections

def mold_image(image, mean_pixel):
	return image.astype(np.float32) - mean_pixel

def compose_image_meta(image_id, original_image_shape, image_shape, window, scale, active_class_ids):
	meta = np.array([image_id] + list(original_image_shape) + list(image_shape) + list(window) + [scale] + 
		list(active_class_ids))
	return meta

def compute_backbone_shapes(backbone_strides, image_shape):
	return np.array([[int(math.ceil(image_shape[0] / stride)), 
		int(math.ceil(image_shape[1] / stride))] for stride in backbone_strides])

def load_image_gt(dataset, image_id, mode, config, augment=False, augmentation=None, use_mini_mask=False):
	image = dataset.load_image(image_id)
	mask, class_ids = dataset.load_mask(image_id)

	original_shape = image.shape
	image, window, scale, padding, crop = utils.resize_image(image, config.image_min_dim, config.image_max_dim, 
		config.image_min_scale, mode)
	mask = utils.resize_mask(mask, scale, padding, crop)

	if augment:
		logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
		if random.randint(0, 1):
			image = np.fliplr(image)
			mask = np.fliplr(mask)

	if augmentation:
		import imgaug
		MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes", "Fliplr", "Flipud", "CropAndPad", "Affine", "PiecewiseAffine"]

		def hook(images, augmenter, parents, default):
			return augmenter.__class__.__name__ in MASK_AUGMENTERS

		image_shape = image.shape
		mask_shape = mask.shape

		det = augmentation.to_deterministic()
		image = det.augment_image(image)
		mask = det.augment_image(mask.astype(np.uint8), hooks=imgaug.HooksImages(activator=hook))

		assert image.shape == image_shape, "Augmentation shouldn't change image size"
		assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"

		mask = mask.astype(np.bool)

	_idx = np.sum(mask, axis=(0, 1)) > 0
	mask = mask[:, :, _idx]
	class_ids = class_ids[_idx]
	bbox = utils.extract_bboxes(mask)

	active_class_ids = np.zeros((dataset.num_classes), dtype=np.int32)
	source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
	active_class_ids[source_class_ids] = 1

	if use_mini_mask:
		mask = utils.minimize_mask(bbox, mask, config.mini_mask_shape)

	image_meta = compose_image_meta(image_id, original_shape, image.shape, window, scale, active_class_ids)

	return image, image_meta, class_ids, bbox, mask

def build_rpn_targets(anchors, gt_class_ids, gt_boxes, config):
	rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
	rpn_bbox = np.zeros((config.rpn_train_anchors_per_image, 4))

	crowd_ix = np.where(gt_class_ids < 0)[0]
	if crowd_ix.shape[0] > 0:
		non_crowd_ix = np.where(gt_class_ids > 0)[0]
		crowd_boxes = gt_boxes[crowd_ix]
		gt_class_ids = gt_class_ids[non_crowd_ix]
		gt_boxes = gt_boxes[non_crowd_ix]
		crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
		crowd_iou_max = np.max(crowd_overlaps, axis=1)
		no_crowd_bool = (crowd_iou_max < 0.001)
	else:
		no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

	overlaps = utils.compute_overlaps(anchors, gt_boxes)
	anchor_iou_argmax = np.argmax(overlaps, axis=1)
	anchor_iou_max = overlaps[np.arange(anchors.shape[0]), anchor_iou_argmax]
	rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1

	gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
	rpn_match[gt_iou_argmax] = 1
	rpn_match[anchor_iou_max >= 0.7] = 1

	ids = np.where(rpn_match == 1)[0]
	extra = len(ids) - (config.rpn_train_anchors_per_image // 2)
	if extra > 0:
		ids = np.random.choice(ids, extra, replace=False)
		rpn_match[ids] = 0

	ids = np.where(rpn_match == -1)[0]
	extra = len(ids) - (config.rpn_train_anchors_per_image - np.sum(rpn_match == 1))
	if extra > 0:
		ids = np.random.choice(ids, extra, replace=False)
		rpn_match[ids] = 0

	ids = np.where(rpn_match == 1)[0]
	ix = 0

	for i, a in zip(ids, anchors[ids]):
		gt = gt_boxes[anchor_iou_argmax[i]]
		gt_h = gt[2] - gt[0]
		gt_w = gt[3] - gt[1]
		gt_center_y = gt[0] + gt_h * 0.5
		gt_center_x = gt[1] + gt_w * 0.5

		a_h = a[2] - a[0]
		a_w = a[3] - a[1]
		a_center_y = a[0] + a_h * 0.5
		a_center_x = a[1] + a_w * 0.5

		rpn_bbox[ix] = [(gt_center_y - a_center_y) / a_h, (gt_center_x - a_center_x) / a_w, np.log(gt_h / a_h), np.log(gt_w / a_w)]
		rpn_bbox[ix] /= config.rpn_bbox_std_dev
		ix += 1

	return rpn_match, rpn_bbox

def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
	rois = np.zeros((count, 4), dtype=np.int32)
	rois_per_box = int(0.9 * count / gt_boxes.shape[0])

	for i in range(gt_boxes.shape[0]):
		gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
		h, w = gt_y2 - gt_y1, gt_x2 - gt_x1

		r_y1, r_y2 = max(gt_y1 - h, 0), min(gt_y2 + h, image_shape[0])
		r_x1, r_x2 = max(gt_x1 - w, 0), min(gt_x2 + w, image_shape[1])

		while True:
			y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
			x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
			threshold = 1
			y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >= threshold][:rois_per_box]
			x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >= threshold][:rois_per_box]
			if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
				break

		y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
		x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
		box_rois = np.hstack([y1, x1, y2, x2])
		rois[rois_per_box * i : rois_per_box * (i + 1)] = box_rois

	remaining_count = count - (rois_per_box * gt_boxes.shape[0])

	while True:
		y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
		x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
		threshold = 1
		y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >= threshold][:remaining_count]
		x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >= threshold][:remaining_count]
		if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
			break

	y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
	x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
	global_rois = np.hstack([y1, x1, y2, x2])
	rois[-remaining_count:] = global_rois
	return rois

def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
	assert rpn_rois.shape[0] > 0
	assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(gt_class_ids.dtype)
	assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(gt_boxes.dtype)
	assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(gt_masks.dtype)

	instance_ids = np.where(gt_class_ids > 0)[0]
	assert instance_ids.shape[0] > 0, "Image must contain instances."
	gt_class_ids = gt_class_ids[instance_ids]
	gt_boxes = gt_boxes[instance_ids]
	gt_masks = gt_masks[:, :, instance_ids]

	rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * (rpn_rois[:, 3] - rpn_rois[:, 1])
	gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
	overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
	for i in range(overlaps.shape[1]):
		overlaps[:, i] = utils.compute_iou(gt_boxes[i], rpn_rois, gt_box_area[i], rpn_roi_area)

	rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
	rpn_roi_iou_max = overlaps[np.arange(overlaps.shape[0]), rpn_roi_iou_argmax]
	rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
	rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

	fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]
	bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

	fg_roi_count = int(config.train_rois_per_image * config.roi_positive_ratio)
	keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False) if fg_ids.shape[0] > fg_roi_count else fg_ids

	remaining = config.train_rois_per_image - keep_fg_ids.shape[0]
	keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False) if bg_ids.shape[0] > remaining else bg_ids

	keep = np.concatenate([keep_fg_ids, keep_bg_ids])
	remaining = config.train_rois_per_image - keep.shape[0]

	if remaining > 0:
		if keep.shape[0] == 0:
			bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
			assert bg_ids.shape[0] >= remaining
			keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
			assert keep_bg_ids.shape[0] == remaining
			keep = np.concatenate([keep, keep_bg_ids])
		else:
			keep_extra_ids = np.random.choice(keep_bg_ids, remaining, replace=True)
			keep = np.concatenate([keep, keep_extra_ids])

	assert keep.shape[0] == config.train_rois_per_image, "keep doesn't match ROI batch size {}, {}".format(keep.shape[0], 
		config.train_rois_per_image)

	rpn_roi_gt_boxes[keep_bg_ids, :] = 0
	rpn_roi_gt_class_ids[keep_bg_ids] = 0

	rois = rpn_rois[keep]
	roi_gt_boxes = rpn_roi_gt_boxes[keep]
	roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
	roi_gt_assignment = rpn_roi_iou_argmax[keep]

	bboxes = np.zeros((config.train_rois_per_image, config.num_classes, 4), dtype=np.float32)
	pos_ids = np.where(roi_gt_class_ids > 0)[0]
	bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(rois[pos_ids], roi_gt_boxes[pos_ids, :4])
	bboxes /= config.bbox_std_dev

	masks = np.zeros((config.train_rois_per_image, config.mask_shape[0], config.mask_shape[1], config.num_classes), dtype=np.float32)

	for i in pos_ids:
		class_id = roi_gt_class_ids[i]
		assert class_id > 0, "class id must be greater than 0"
		gt_id = roi_gt_assignment[i]
		class_mask = gt_masks[:, :, gt_id]

		if config.use_mini_mask:
			placeholder = np.zeros(config.image_shape[:2], dtype=bool)
			gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
			gt_h, gt_w = gt_y2 - gt_y1, gt_x2 - gt_x1
			placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
			class_mask = placeholder

		y1, x1, y2, x2 = rois[i].astype(np.int32)
		m = class_mask[y1:y2, x1:x2]
		mask = utils.resize(m, config.mask_shape)
		masks[i, :, :, class_id] = mask

	return rois, roi_gt_class_ids, bboxes, masks

def data_generator(dataset, mode, config, shuffle=True, augment=False, augmentation=None, random_rois=0, batch_size=1, 
	detection_targets=False, no_augmentation_sources=None):
	b, image_index, image_ids, error_count = 0, -1, np.copy(dataset.image_ids), 0
	no_augmentation_sources = no_augmentation_sources or []

	backbone_shapes = compute_backbone_shapes(config.backbone_strides, config.image_shape)
	anchors = utils.generate_pyramid_anchors(config.rpn_anchor_scales, config.rpn_anchor_ratios, backbone_shapes, 
		config.backbone_strides)

	while True:
		try:
			image_index = (image_index + 1) % len(image_ids)
			if shuffle and image_index == 0:
				np.random.shuffle(image_ids)
			image_id = image_ids[image_index]
			if dataset.image_info[image_id]["source"] in no_augmentation_sources:
				image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(dataset, image_id, mode, config, 
					augment=augment, augmentation=None, use_mini_mask=config.use_mini_mask)
			else:
				image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(dataset, image_id, mode, config, 
					augment=augment, augmentation=augmentation, use_mini_mask=config.use_mini_mask)

			if not np.any(gt_class_ids > 0):
				continue

			rpn_match, rpn_bbox = build_rpn_targets(anchors, gt_class_ids, gt_boxes, config)

			if random_rois:
				rpn_rois = generate_random_rois(image.shape, random_rois, gt_class_ids, gt_boxes)
				if detection_targets:
					rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, 
						gt_masks, config)

			if b == 0:
				batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
				batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
				batch_rpn_bbox = np.zeros([batch_size, config.rpn_train_anchors_per_image, 4], dtype=rpn_bbox.dtype)
				batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
				batch_gt_class_ids = np.zeros([batch_size, config.max_gt_instances], dtype=np.int32)
				batch_gt_boxes = np.zeros([batch_size, config.max_gt_instances, 4], dtype=np.int32)
				batch_gt_masks = np.zeros([batch_size, gt_masks.shape[0], gt_masks.shape[1], config.max_gt_instances], 
					dtype=gt_masks.dtype)
				if random_rois:
					batch_rpn_rois = np.zeros([batch_size, rpn_rois.shape[0], 4], dtype=rpn_rois.dtype)
					if detection_targets:
						batch_rois = np.zeros((batch_size,) + rois.shape, dtype=rois.dtype)
						batch_mrcnn_class_ids = np.zeros((batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
						batch_mrcnn_bbox = np.zeros((batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
						batch_mrcnn_mask = np.zeros((batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

			if gt_boxes.shape[0] > config.max_gt_instances:
				ids = np.random.choice(np.arange(gt_boxes.shape[0]), config.max_gt_instances, replace=False)
				gt_class_ids = gt_class_ids[ids]
				gt_boxes = gt_boxes[ids]
				gt_masks = gt_masks[:, :, ids]

			batch_image_meta[b] = image_meta
			batch_rpn_match[b] = rpn_match[:, np.newaxis]
			batch_rpn_bbox[b] = rpn_bbox
			batch_images[b] = mold_image(image.astype(np.float32), config.mean_pixel)
			batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
			batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
			batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
			if random_rois:
				batch_rpn_rois[b] = rpn_rois
				if detection_targets:
					batch_rois[b] = rois
					batch_mrcnn_class_ids[b] = mrcnn_class_ids
					batch_mrcnn_bbox[b] = mrcnn_bbox
					batch_mrcnn_mask[b] = mrcnn_mask
			b += 1

			if b >= batch_size:
				inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, 
				batch_gt_boxes, batch_gt_masks]
				outputs = []
				if random_rois:
					inputs.extend([batch_rpn_rois])
					if detection_targets:
						inputs.extend([batch_rois])
						batch_mrcnn_class_ids = np.expand_dims(batch_mrcnn_class_ids, -1)
						outputs.extend([batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])
				yield inputs, outputs
				b = 0

		except(GeneratorExit, KeyboardInterrupt):
			raise
		except:
			logging.exception("Error processing image {}".format(dataset.image_info[image_id]))
			error_count += 1
			if error_count > 5:
				raise


class BatchNorm(KL.BatchNormalization):
	def call(self, inputs, training=None):
		return super(self.__class__, self).call(inputs, training=training)

class ProposalLayer(KE.Layer):
	def __init__(self, batch_size, proposal_count, config, **kwargs):
		super().__init__(**kwargs)
		self.batch_size = batch_size
		self.proposal_count = proposal_count
		self.config = config

	def call(self, inputs):
		scores, deltas, anchors = inputs[0][:, :, 1], inputs[1], inputs[2]
		deltas = deltas * np.reshape(self.config.rpn_bbox_std_dev, [1, 1, 4])

		pre_nms_limit = tf.minimum(self.config.pre_nms_limit, tf.shape(anchors)[1])
		ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices

		scores = utils.batch_slic([scores, ix], lambda x, y: tf.gather(x, y), self.batch_size)
		deltas = utils.batch_slic([deltas, ix], lambda x, y: tf.gather(x, y), self.batch_size)
		pre_nms_anchors = utils.batch_slic([anchors, ix], lambda x, y: tf.gather(x, y), self.batch_size, 
			name=["pre_nms_anchors"])
		boxes = utils.batch_slic([pre_nms_anchors, deltas], lambda x, y: apply_box_deltas_graph(x, y), self.batch_size, 
			name=["refined_anchors"])
		boxes = utils.batch_slic([boxes], lambda x: clip_boxes_graph(x, np.array([0, 0, 1, 1], dtype=np.float32)), 
			self.batch_size, name=["refined_anchors_clipped"])

		def nms(boxes, scores):
			indices = tf.image.non_max_suppression(boxes, scores, self.proposal_count, self.config.nms_threshold, 
				name="rpn_non_max_suppression")
			proposals = tf.gather(boxes, indices)
			padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
			proposals = tf.pad(proposals, [(0, padding), (0, 0)])
			return proposals

		proposals = utils.batch_slic([boxes, scores], nms, self.batch_size)

		return proposals

	def compute_output_shape(self, input_shape):
		return (None, self.proposal_count, 4)

class DetectionTargetLayer(KE.Layer):
	def __init__(self, batch_size, config, **kwargs):
		super().__init__(**kwargs)
		self.batch_size = batch_size
		self.config = config

	def call(self, inputs):
		proposals = inputs[0]
		gt_class_ids = inputs[1]
		gt_boxes = inputs[2]
		gt_masks = inputs[3]
		name = ["rois", "target_class_ids", "target_bbox", "target_mask"]

		outputs = utils.batch_slic([proposals, gt_class_ids, gt_boxes, gt_masks], 
			lambda w, x, y, z: detection_targets_graph(w, x, y, z, self.config), self.batch_size, name=name)

		return outputs

	def compute_output_shape(self, input_shape):
		return [(None, self.config.train_rois_per_image, 4),
		(None, self.config.train_rois_per_image),
		(None, self.config.train_rois_per_image, 4),
		(None, self.config.train_rois_per_image, self.config.mask_shape[0], self.config.mask_shape[1])]

	def compute_mask(self, inputs, mask=None):
		return [None, None, None, None]

class DetectionLayer(KE.Layer):
	def __init__(self, batch_size, config, **kwargs):
		super().__init__(**kwargs)
		self.batch_size = batch_size
		self.config = config

	def call(self, inputs):
		rois, mrcnn_class, mrcnn_bbox, image_meta = inputs[0], inputs[1], inputs[2], inputs[3]
		m = parse_image_meta_graph(image_meta)
		image_shape = m["image_shape"][0]
		window = norm_boxes_graph(m["window"], image_shape[:2])
		detections_batch = utils.batch_slic([rois, mrcnn_class, mrcnn_bbox, window], lambda x, y, z, w: refine_detections_graph(x, 
			y, z, w, self.config), self.batch_size)
		return tf.reshape(detections_batch, [self.batch_size, self.config.detection_max_instances, 6])

	def compute_output_shape(self, input_shape):
		return (None, self.config.detection_max_instances, 6)

class PyramidROIAlign(KE.Layer):
	def __init__(self, pool_shape, **kwargs):
		super().__init__(**kwargs)
		self.pool_shape = tuple(pool_shape)

	def call(self, inputs):
		boxes, image_meta, feature_maps = inputs[0], inputs[1], inputs[2:]

		y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
		h, w = y2 - y1, x2 - x1

		image_shape = parse_image_meta_graph(image_meta)["image_shape"][0]
		image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)

		roi_level = log2_graph(tf.math.sqrt(h * w) / (224.0 / tf.math.sqrt(image_area)))
		roi_level = tf.math.minimum(5, tf.math.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
		roi_level = tf.squeeze(roi_level, 2)

		pooled, box_to_level = [], []
		for i, level in enumerate(range(2, 6)):
			ix = tf.where(tf.math.equal(roi_level, level))
			level_boxes = tf.gather_nd(boxes, ix)
			box_indices = tf.cast(ix[:, 0], tf.int32)
			box_to_level.append(ix)
			level_boxes = tf.stop_gradient(level_boxes)
			box_indices = tf.stop_gradient(box_indices)
			pooled.append(tf.image.crop_and_resize(feature_maps[i], level_boxes, box_indices, self.pool_shape, method="bilinear"))

		pooled = tf.concat(pooled, axis=0)
		box_to_level = tf.concat(box_to_level, axis=0)
		box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
		box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

		sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
		ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
		ix = tf.gather(box_to_level[:, 2], ix)
		pooled = tf.gather(pooled, ix)

		shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
		pooled = tf.reshape(pooled, shape)

		return pooled

	def compute_output_shape(self, input_shape):
		return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)

class MaskRCNN():
	def __init__(self, batch_size, mode, config):
		self.batch_size = batch_size
		self.mode = mode
		self.config = config
		self._anchor_cache = {}
		self.keras_model = self.build(self.mode, self.config)
		self.log_dir = "D:/program/mask_rcnn/log"
		self.checkpoint_path = "D:/program/mask_rcnn/checkpoint/mask_rcnn_model.h5"

	def get_anchors(self, config):
		backbone_shapes = np.array([[int(math.ceil(config.image_shape[0] / stride)), int(math.ceil(config.image_shape[1] / stride))] 
			for stride in config.backbone_strides])

		if not tuple(config.image_shape) in self._anchor_cache:
			a = utils.generate_pyramid_anchors(config.rpn_anchor_scales, config.rpn_anchor_ratios, backbone_shapes, 
				config.backbone_strides)
			self.anchors = a
			self._anchor_cache[tuple(config.image_shape)] = utils.norm_boxes(a, config.image_shape[:2])

		return self._anchor_cache[tuple(config.image_shape)]

	def build(self, mode, config):
		assert mode in ["training", "inference"]
		h, w = config.image_shape[:2]
		if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
			raise Exception("Image size must be dividable by 2 at least 6 times.")

		input_image = KL.Input(shape=[None, None, config.image_shape[2]], name="input_image")
		input_image_meta = KL.Input(shape=[config.image_meta_size], name="input_image_meta")

		if mode == "training":
			input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
			input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)
			input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
			input_gt_boxes = KL.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
			gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_gt_boxes)
			if config.use_mini_mask:
				input_gt_masks = KL.Input(shape=config.mini_mask_shape + [None], name="input_gt_masks", dtype=bool)
			else:
				input_gt_masks = KL.Input(shape=[config.image_shape[0],config.image_shape[1],None], name="input_gt_masks", dtype=bool)
		else:
			input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

		_, C2, C3, C4, C5 = resnet_graph(input_image, config.architecture, stage5=True, train_bn=config.train_bn)

		P5 = KL.Conv2D(256, (1, 1), name="fpn_c5p5")(C5)
		P4 = KL.Add(name="fpn_p4add")([KL.UpSampling2D(size=(2, 2), name="fpn_p5unsampled")(P5), 
			KL.Conv2D(256, (1, 1), name="fpn_c4p4")(C4)])
		P3 = KL.Add(name="fpn_p3add")([KL.UpSampling2D(size=(2, 2), name="fpn_p4unsampled")(P4), 
			KL.Conv2D(256, (1, 1), name="fpn_c3p3")(C3)])
		P2 = KL.Add(name="fpn_p2add")([KL.UpSampling2D(size=(2, 2), name="fpn_p3unsampled")(P3), 
			KL.Conv2D(256, (1, 1), name="fpn_c2p2")(C2)])

		P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
		P3 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
		P4 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
		P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
		P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

		rpn_feature_maps = [P2, P3, P4, P5, P6]
		mrcnn_feature_maps = [P2, P3, P4, P5]

		if mode == "training":
			anchors = self.get_anchors(config)
			anchors = np.broadcast_to(anchors, (self.batch_size,) + anchors.shape)
			anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
		else:
			anchors = input_anchors

		layer_outputs = []
		rpn = build_rpn_model(len(config.rpn_anchor_ratios), config.top_down_pyramid_size)
		for P in rpn_feature_maps:
			layer_outputs.append(rpn([P]))

		output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
		outputs = list(zip(*layer_outputs))
		outputs = [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]
		rpn_class_logits, rpn_class, rpn_bbox = outputs

		proposal_count = config.post_nms_rois_training if self.mode == "training" else config.post_nms_rois_inference
		rpn_rois = ProposalLayer(self.batch_size, proposal_count, config, name="ROI")([rpn_class, rpn_bbox, anchors])

		if mode == "training":
			active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)
			target_rois = rpn_rois

			rois,target_class_ids,target_bbox,target_mask = DetectionTargetLayer(self.batch_size, config, 
				name="proposal_targets")([target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

			mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta, 
				config.pool_size, config.num_classes, config.train_bn, config.fpn_class_fc_layers_size)

			mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps, input_image_meta, config.mask_pool_size, config.num_classes, 
				config.train_bn)

			output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

			rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")([input_rpn_match, 
				rpn_class_logits])
			rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(self.batch_size, *x), name="rpn_bbox_loss")([input_rpn_bbox, 
				input_rpn_match, rpn_bbox])
			class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")([target_class_ids, 
				mrcnn_class_logits, active_class_ids])
			bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")([target_bbox, target_class_ids, 
				mrcnn_bbox])
			mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")([target_mask, target_class_ids, 
				mrcnn_mask])

			inputs = [input_image,input_image_meta,input_rpn_match,input_rpn_bbox,input_gt_class_ids,input_gt_boxes,input_gt_masks]
			outputs = [rpn_class_logits, rpn_class, rpn_bbox, mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, 
			output_rois, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]

			model = KM.Model(inputs, outputs, name="mask_rcnn")

		else:
			mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta, 
				config.pool_size, config.num_classes, config.train_bn, config.fpn_class_fc_layers_size)

			detections = DetectionLayer(self.batch_size, config, name="mrcnn_detection")([rpn_rois, mrcnn_class, mrcnn_bbox, 
				input_image_meta])
				
			detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)

			mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps, input_image_meta, config.mask_pool_size, 
				config.num_classes, config.train_bn)

			model = KM.Model([input_image, input_image_meta, input_anchors], [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, 
				rpn_rois, rpn_class, rpn_bbox], name="mask_rcnn")

		return model

	def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
		if verbose > 0 and keras_model is None:
			log("Selecting layers to train")
		keras_model = keras_model or self.keras_model
		layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") else keras_model.layers

		for layer in layers:
			if layer.__class__.__name__ == "Model":
				print("In model: ", layer.name)
				self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
				continue
			if not layer.weights:
				continue
			trainable = bool(re.fullmatch(layer_regex, layer.name))
			if layer.__class__.__name__ == "TimeDistributed":
				layer.layer.trainable = trainable
			else:
				layer.trainable = trainable
			if trainable and verbose > 0:
				log("{}{:20}   ({})".format(" " * indent, layer.name, layer.__class__.__name__))

	def compile(self, learning_rate, momentum):
		optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, clipnorm=self.config.gradient_clip_norm)
		self.keras_model._losses = []
		self.keras_model._per_input_losses = {}
		loss_names = ["rpn_class_loss", "rpn_bbox_loss", "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]

		for name in loss_names:
			layer = self.keras_model.get_layer(name)
			if layer.output in self.keras_model.losses:
				continue
			loss = (tf.math.reduce_mean(layer.output, keep_dims=True) * self.config.loss_weights.get(name, 1.))
			self.keras_model.add_loss(loss)

		reg_losses = [keras.regularizers.l2(self.config.weight_decay)(w) / tf.cast(tf.size(w), tf.float32) 
		for w in self.keras_model.trainable_weights if "gamma" not in w.name and "beta" not in w.name]
		self.keras_model.add_loss(tf.math.add_n(reg_losses))
		self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

		for name in loss_names:
			if name in self.keras_model.metrics_names:
				continue
			layer = self.keras_model.get_layer(name)
			self.keras_model.metrics_names.append(name)
			loss = (tf.math.reduce_mean(layer.output, keep_dims=True) * self.config.loss_weights.get(name, 1.))
			self.keras_model.metrics_tensors.append(loss)

	def train(self, train_dataset, val_dataset, learning_rate, epochs, layers, augmentation=None, 
		custom_callbacks=None, no_augmentation_sources=None):
		assert self.mode == "training", "Create model in training mode"
		layer_regex = {"heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
		"3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
		"4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
		"5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
		"all": ".*"}
		if layers in layer_regex.keys():
			layers = layer_regex[layers]

		train_generator = data_generator(train_dataset, self.config.image_resize_mode, self.config, shuffle=True, 
			augmentation=augmentation, batch_size=self.batch_size, no_augmentation_sources=no_augmentation_sources)
		val_generator = data_generator(val_dataset, self.config.image_resize_mode, self.config, shuffle=True, 
			batch_size=self.batch_size)

		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)

		callbacks = [keras.callbacks.TensorBoard(self.log_dir, histogram_freq=0, write_graph=True, write_images=False), 
		keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_weights_only=True)]

		if custom_callbacks:
			callbacks += custom_callbacks

		log("\nStarting at LR={}\n".format(learning_rate))
		log("Checkpoint Path: {}".format(self.checkpoint_path))
		self.set_trainable(layers)
		self.compile(learning_rate, self.config.learning_momentum)

		self.keras_model.fit_generator(
			train_generator,
			initial_epoch=self.epoch,
			epochs=epochs,
			steps_per_epoch=self.config.steps_per_epoch,
			callbacks=callbacks,
			validation_data=val_generator,
			validation_steps=self.config.validation_steps,
			max_queue_size=100,
			workers=1,
			use_multiprocessing=False)

		self.epoch = max(self.epoch, epochs)

	def mold_inputs(self, images):
		molded_images, image_metas, windows = [], [], []
		for image in images:
			molded_image, window, scale, padding, crop = utils.resize_image(image, self.config.image_min_dim, 
				self.config.image_max_dim, self.config.image_min_scale, self.config.image_resize_mode)
			molded_image = mold_image(molded_image, self.config.mean_pixel)
			image_meta = compose_image_meta(0, image.shape, molded_image.shape, window, scale, 
				np.zeros([self.config.num_classes], dtype=np.int32))
			molded_images.append(molded_image)
			image_metas.append(image_meta)
			windows.append(window)
		molded_images = np.stack(molded_images)
		image_metas = np.stack(image_metas)
		windows = np.stack(windows)
		return molded_images, image_metas, windows

	def unmold_detections(self, detections, mrcnn_mask, original_image_shape, image_shape, window):
		zero_ix = np.where(detections[:, 4] == 0)[0]
		N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

		boxes = detections[:N, :4]
		class_ids = detections[:N, 4].astype(np.int32)
		scores = detections[:N, 5]
		masks = mrcnn_mask[np.arange(N), :, :, class_ids]

		window = utils.norm_boxes(window, image_shape[:2])
		wy1, wx1, wy2, wx2 = window
		shift = np.array([wy1, wx1, wy1, wx1])
		wh, ww = wy2 - wy1, wx2 - wx1
		scale = np.array([wh, ww, wh, ww])
		boxes = np.divide(boxes - shift, scale)
		boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

		exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
		if exclude_ix.shape[0] > 0:
			boxes = np.delete(boxes, exclude_ix, axis=0)
			class_ids = np.delete(class_ids, exclude_ix, axis=0)
			scores = np.delete(scores, exclude_ix, axis=0)
			masks = np.delete(masks, exclude_ix, axis=0)
			N = class_ids.shape[0]

		full_masks = []
		for i in range(N):
			full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
			full_masks.append(full_mask)
		full_masks = np.stack(full_masks, axis=-1) if full_masks else np.empty(original_image_shape + (0,))

		return boxes, class_ids, scores, full_masks

	def detect(self, images, verbose=0):
		assert self.mode == "inference", "Create model in inference mode"
		assert len(images) == self.batch_size, "len(images) must be equal to batch size"

		if verbose:
			log("Processing {} images".format(len(images)))
			for image in images:
				log("image", image)

		molded_images, image_metas, windows = self.mold_inputs(images)
		image_shape = molded_images[0].shape

		for g in molded_images[1:]:
			assert g.shape == image_shape, "After resizing, all images should have the same size"

		anchors = self.get_anchors(self.config)
		anchors = np.broadcast_to(anchors, (self.batch_size,) + anchors.shape)

		if verbose:
			log("molded images", molded_images)
			log("image_metas", image_metas)
			log("anchors", anchors)

		detections, _, _, mrcnn_mask, _, _, _ = self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)

		results = []
		for i, image in enumerate(images):
			final_rois, final_class_ids, final_scores, final_masks = self.unmold_detections(detections[i], mrcnn_mask[i], 
				image.shape, molded_images[i].shape, windows[i])
			results.append({"rois": final_rois, "class_ids": final_class_ids, "scores": final_scores, "masks": final_masks})

		return results

	def load_weights(self, file_path, by_name=False, exclude=None):
		if exclude:
			by_name = True
		f = h5py.File(file_path, mode="r")
		if "layer_names" not in f.attrs and "model_weights" in f:
			f = f["model_weights"]
		keras_model = self.keras_model

		layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") else keras_model.layers
		if exclude:
			layers = filter(lambda l: l.name not in exclude, layers)
		if by_name:
			saving.load_weights_from_hdf5_group_by_name(f, layers)
		else:
			saving.load_weights_from_hdf5_group(f, layers)
		if hasattr(f, "close"):
			f.close()
		self.epoch = 0
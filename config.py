import numpy as np


class Config():
	name = None
	num_classes = 1
	architecture = "resnet101"
	train_bn = False
	use_mini_mask = True
	backbone_strides = [4, 8, 16, 32, 64]
	rpn_anchor_scales = [32, 64, 128, 256, 512]
	rpn_anchor_ratios = [0.5, 1.0, 2.0]
	top_down_pyramid_size = 256
	rpn_train_anchors_per_image = 256
	post_nms_rois_training = 2000
	post_nms_rois_inference = 1000
	nms_threshold = 0.7
	rpn_bbox_std_dev = np.array([0.1, 0.1, 0.2, 0.2])
	bbox_std_dev = np.array([0.1, 0.1, 0.2, 0.2])
	train_rois_per_image = 200
	max_gt_instances = 100
	roi_positive_ratio = 0.33
	pre_nms_limit = 6000
	mini_mask_shape = [56, 56]
	mask_shape = [28, 28]
	pool_size = 7
	fpn_class_fc_layers_size = 1024
	mask_pool_size = 14
	detection_min_confidence = 0.7
	detection_max_instances = 100
	detection_nms_threshold = 0.3
	image_min_dim = 800
	image_max_dim = 1024
	image_min_scale = 0
	image_resize_mode = "square"
	mean_pixel = np.array([123.7, 116.8, 103.9])
	gradient_clip_norm = 5.0
	weight_decay = 0.0001
	learning_rate = 0.001
	learning_momentum = 0.9
	steps_per_epoch = 1000
	validation_steps = 50
	loss_weights = {"rpn_class_loss": 1., "rpn_bbox_loss": 1., "mrcnn_class_loss": 1., 
	"mrcnn_bbox_loss": 1., "mrcnn_mask_loss": 1.}

	def __init__(self):
		if self.image_resize_mode == "crop":
			self.image_shape = np.array([self.image_min_dim, self.image_min_dim, 3])
		else:
			self.image_shape = np.array([self.image_max_dim, self.image_max_dim, 3])
		self.image_meta_size = 1 + 3 + 3 + 4 +1 + self.num_classes
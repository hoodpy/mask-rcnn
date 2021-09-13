from keras.utils.data_utils import get_file

TF_WEIGHTS_PATH_NO_TOP = "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_path = get_file("D:/program/mask_rcnn/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
	TF_WEIGHTS_PATH_NO_TOP, cache_subdir="D:/program/mask_rcnn/models", md5_hash='a268eb855778b3df3c7506639542a6af')
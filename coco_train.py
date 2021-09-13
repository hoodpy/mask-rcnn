import os
import time
import numpy as np
import imgaug
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import zipfile
import urllib.request
import shutil
from config import Config
from model import MaskRCNN
import utils


def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
	if rois is None:
		return []

	results = []
	for image_id in image_ids:
		for i in range(rois.shape[0]):
			class_id = class_ids[i]
			score = scores[i]
			bbox = np.around(rois[i], 1)
			mask = masks[:, :, i]
			result = {"image_id": image_id, "category_id": dataset.get_source_class_id(class_id, "coco"),
			"bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]], "score": score,
			"segmentation": maskUtils.encode(np.asfortranarray(mask))}
			results.append(result)

	return results

def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
	image_ids = image_ids or dataset.image_ids()

	if limit:
		image_ids = image_ids[:limit]

	coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]
	t_prediction = 0
	t_start = time.time()

	results = []
	for i, image_id in enumerate(image_ids):
		image = dataset.load_image(image_id)
		t = time.time()
		r = model.detect([image], verbose=0)[0]
		t_prediction += (time.time() - t)
		image_results = build_coco_results(dataset, coco_image_ids[i : i+1], r["rois"], r["class_ids"], r["scores"], 
			r["masks"].astype(np.uint8))
		results.extend(image_results)

	coco_results = coco.loadRes(results)

	cocoEval = COCOeval(coco, coco_results, eval_type)
	cocoEval.params.imgIds = coco_image_ids
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()

	print("Prediction time: {}. Average {}/image".format(t_prediction, t_prediction / len(image_ids)))
	print("Total time: " + str(time.time() - t_start))


class CocoConfig(Config):
	name = "coco"
	num_classes = 81
	
class CocoDataset(utils.Dataset):
	def load_coco(self, dataset_dir, subset, year="2014", class_ids=None, class_map=None, return_coco=False, auto_download=False):
		if auto_download:
			self.auto_download(dataset_dir, subset, year)

		coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))

		if subset == "minival" or subset == "valminusminival":
			subset = "val"
		image_dir = "{}/{}{}".format(dataset_dir, subset, year)

		if not class_ids:
			class_ids = sorted(coco.getCatIds())

		if class_ids:
			image_ids = []
			for id in class_ids:
				image_ids.extend(list(coco.getImgIds(catIds=[id])))
			image_ids = list(set(image_ids))
		else:
			image_ids = list(coco.imgs.keys())

		for i in class_ids:
			self.add_class("coco", i, coco.loadCats(i)[0]["name"])

		for i in image_ids:
			self.add_image("coco", image_id=i, path=os.path.join(image_dir, coco.imgs[i]["file_name"]), width=coco.imgs[i]["width"], 
				height=coco.imgs[i]["height"], annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)))

		if return_coco:
			return coco

	def auto_download(self, dataDir, dataType, dataYear):
		if dataType == "minival" or dataType == "valminusminival":
			imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
			imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
			imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
		else:
			imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
			imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
			imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)

		if not os.path.exists(dataDir):
			os.makedirs(dataDir)

		if not os.path.exists(imgDir):
			os.makedirs(imgDir)
			print("Downloading images to " + imgZipFile + " ...")
			with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, "wb") as out:
				shutil.copyfileobj(resp, out)
			print("... done downloading.")
			print("Unzipping " + imgZipFile)
			with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
				zip_ref.extractall(imgDir)
			print("... done unzipping")
		print("Will use images in " + imgDir)

		annDir = "{}/annotations".format(dataDir)
		if dataType == "minival":
			annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
			annFile = "{}/instances_minival2014.json".format(annDir)
			annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
			unZipDir = annDir
		elif dataType == "valminusminival":
			annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
			annFile = "{}/instances_valminusminival2014.json".format(annDir)
			annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
			unZipDir = annDir
		else:
			annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
			annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
			annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
			unZipDir = dataDir

		if not os.path.exists(annDir):
			os.makedirs(annDir)
		if not os.path.exists(annFile):
			if not os.path.exists(annZipFile):
				print("Downloading zipped annotations to " + annZipFile + " ...")
				with urllib.request.urlopen(annURL) as resp, open(annZipFile, "wb") as out:
					shutil.copyfileobj(resp, out)
				print("... done downloading.")
			print("Unzipping " + annZipFile)
			with zipfile.ZipFile(annZipFile, "r") as zip_ref:
				zip_ref.extractall(unZipDir)
			print("... done unzipping")
		print("Will use annotations in " + annFile)

	def load_mask(self, image_id):
		image_info = self.image_info[image_id]
		if image_info["source"] != "coco":
			return super(CocoDataset, self).load_mask(image_id)

		instance_masks, class_ids = [], []
		annotations = self.image_info[image_id]["annotations"]

		for annotation in annotations:
			class_id = self.map_source_class_id("coco.{}".format(annotation["category_id"]))
			if class_id:
				m = self.annToMask(annotation, image_info["height"], image_info["width"])
				if m.max() < 1:
					continue
				if annotation["iscrowd"]:
					class_id *= -1
					if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
						m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
				instance_masks.append(m)
				class_ids.append(class_id)

		if class_ids:
			mask = np.stack(instance_masks, axis=2).astype(np.bool)
			class_ids = np.array(class_ids, dtype=np.int32)
			return mask, class_ids
		else:
			return super(CocoDataset, self).load_mask(image_id)

	def image_reference(self, image_id):
		info = self.image_info[image_id]
		if info["source"] == "coco":
			return "http://cocodataset.org/#explore?id={}".format(info["id"])
		else:
			super(CocoDataset, self).image_reference(image_id)

	def annToRLE(self, ann, height, width):
		segm = ann["segmentation"]
		if isinstance(segm, list):
			rles = maskUtils.frPyObjects(segm, height, width)
			rle = maskUtils.merge(rles)
		elif isinstance(segm["counts"], list):
			rle = maskUtils.frPyObjects(segm, height, width)
		else:
			rle = ann["segmentation"]
		return rle

	def annToMask(self, ann, height, width):
		rle = self.annToRLE(ann, height, width)
		m = maskUtils.decode(rle)
		return m


if __name__ == "__main__":
	command = "train"
	dataset = "E:/COCO"
	year = "2014"
	model_path = "D:/program/mask_rcnn/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
	limit = 500
	download = False

	class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

	if command == "train":
		config = CocoConfig()
		batch_size = 2
	else:
		class InferenceConfig(CocoConfig):
			detection_min_confidence = 0
		config = InferenceConfig()
		batch_size = 1

	if command == "train":
		model = MaskRCNN(batch_size, mode="training", config=config)
	else:
		model = MaskRCNN(batch_size, mode="inference", config=config)

	model.load_weights(model_path, by_name=True)
	print("Loading weights ", model_path)

	if command == "train":
		dataset_train = CocoDataset()
		dataset_train.load_coco(dataset, "train", year=year, auto_download=download)
		if year in "2014":
			dataset_train.load_coco(dataset, "valminusminival", year=year, auto_download=download)
		dataset_train.prepare()

		dataset_val = CocoDataset()
		val_type = "val" if year in "2017" else "minival"
		dataset_val.load_coco(dataset, val_type, year, auto_download=download)
		dataset_val.prepare()

		augmentation = imgaug.augmenters.Fliplr(0.5)

		print("Training network heads")
		model.train(dataset_train, dataset_val, config.learning_rate, epochs=40, layers="heads", augmentation=augmentation)

		print("Fine tune Resnet stage 4 and up")
		model.train(dataset_train, dataset_val, config.learning_rate, epochs=120, layers="4+", augmentation=augmentation)

		print("Fine tune all layers")
		model.train(dataset_train, dataset_val, config.learning_rate / 10, epochs=160, layers="all", augmentation=augmentation)

	elif command == "evaluate":
		dataset_val = CocoDataset()
		val_type = "val" if year in "2017" else "minival"
		coco = dataset_val.load_coco(dataset, val_type, year, return_coco=True, auto_download=download)
		dataset_val.prepare()
		print("Running COCO evaluation on {} images.".format(limit))
		evaluate_coco(model, dataset_val, coco, "bbox", limit)

	else:
		print("'{}' is not recognized. Use 'train' or 'evaluate'".format(command))
import torchreid
from torchvision.io import read_image, ImageReadMode
import torch
import os
import pickle
import cv2
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms


def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


class SingleQueryModel:
    def __init__(self, model_name, weight_path, query_path):
        self.model_name = model_name
        self.model = None
        self.is_torchreid = False
        self.weight_path = weight_path
        self.query_path = query_path
        self.euclidean = False
        self.initialize_model()

    """
    :param test_path: string path to new test image
    """

    @staticmethod
    def euclidean_squared_distance(input1, input2):
        """Computes euclidean squared distance.

        Args:
            input1 (torch.Tensor): 2-D feature matrix.
            input2 (torch.Tensor): 2-D feature matrix.

        Returns:
            torch.Tensor: distance matrix.
        """
        m, n = input1.size(0), input2.size(0)
        mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
        mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat = mat1 + mat2
        distmat.addmm_(input1, input2.t(), beta=1, alpha=-2)
        return distmat

    @staticmethod
    def cosine_distance(input1, input2):
        """Computes cosine distance.

        Args:
            input1 (torch.Tensor): 2-D feature matrix.
            input2 (torch.Tensor): 2-D feature matrix.

        Returns:
            torch.Tensor: distance matrix.
        """
        input1_normed = F.normalize(input1, p=2, dim=1)
        input2_normed = F.normalize(input2, p=2, dim=1)
        distmat = 1 - torch.mm(input1_normed, input2_normed.t())
        return distmat

    @staticmethod
    def simple_cosine(input1, input2):
        return 1 - np.matmul(np.squeeze(input1), np.squeeze(input2).transpose())

    def test_image(self, test_path):
        query = self.load_image(self.query_path)
        gallery = self.load_image(test_path)
        return self.run(query, gallery)

    def test_matrix(self, img_matrix):
        query = self.load_image(self.query_path)
        if not self.is_torchreid:
            return self.run(query, np.array(img_matrix))
        else:
            transform = transforms.ToTensor()
            test_tensor = transform(img_matrix)
            return self.run(query, test_tensor)

    def initialize_model(self):
        if self.is_torchreid:
            model = torchreid.models.build_model(
                name=self.model_name,
                num_classes=2,  # number of training identities, so might not matter
                loss="softmax",
                pretrained=True
            )
            torchreid.utils.load_pretrained_weights(model, self.weight_path)
            model.eval()  # don't want to train
            self.model = model
        else:
            self.model = cv2.dnn.readNet("pretrained/youtu_reid_baseline_lite.onnx")

    def process(self, img):
        if self.is_torchreid:
            with torch.no_grad():
                return self.model(img.float())
        else:
            #print(img.shape)
            blob = cv2.dnn.blobFromImage(img, 0.5, (128, 256), (128, 128, 128), False, False)
            #print(blob.shape)
            self.model.setInput(blob)
            res = self.model.forward()
           # print(res.shape)
            return cv2.normalize(res, None)


    """
    :param path: string path to image
    :return: Tensor[image_channels, image_height, image_width]
    """

    def load_image(self, path):
        if not self.is_torchreid:
            img = cv2.imread(path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            return read_image(path, mode=ImageReadMode.RGB)  # force as RGB

    def distance_metric(self, input1, input2):
        if not self.is_torchreid:
            return self.simple_cosine(input1, input2)
        elif self.euclidean:
            return self.euclidean_squared_distance(input1, input2)
        else:
            return self.cosine_distance(input1, input2)

    def run(self, query, gallery):
        # 1.) import query and gallery img as tensors
        # add batch dimension (N=1)
        if self.is_torchreid:
            query = query.unsqueeze(0)
            gallery = gallery.unsqueeze(0)

        # 2.) run model on tensor inputs
        query_features = self.process(query)
        gallery_features = self.process(gallery)
        # print(query_features)
        # print(gallery_features)
        # not sure what this does, but scared what happens if I don't
        # query_features = query_features.cpu()
        # gallery_features = gallery_features.cpu()
        # 3.) compute distance. Before, was distance matrix, but also had many queries and galleries, so unnecessary.

        dist = self.distance_metric(query_features, gallery_features)
        if self.is_torchreid:
            dist = dist.numpy()
            dist = dist[0][0]
        return dist
        # correct: 22522.7500
        # wrong: 70822.5000
        # 4.) Somehow use distance value and determine if counts as a match or not


if __name__ == "__main__":
    # 0_5: [0.0072414875, 0.07200009, 0.05656421, 0.009953499, 0.08393031]
    # 1_0: [0.046495795, 0.13216895, 0.41989136, 0.07390034, 0.48588467]
    # ain 1_0: [0.024422288, 0.099429786, 0.18903148, 0.011394441, 0.56568325]
    # ibn 1_0: [0.04042691, 0.57059103, 0.6661659, 0.3205316, 0.5463598]

    MODEL = "osnet_x0_5"
    directory = 'single_query_test/test'
    #directory = 'reid-data/grid/underground_reid/probe'
    TEST_PATHS = get_all_file_paths(directory)
    WEIGHT_PATH = "pretrained/osnet_x0_5_imagenet.pth"
    #QUERY_PATH = 'reid-data/grid/underground_reid/probe/0005_2_25100_229_94_99_249.jpeg'
    QUERY_PATH = "single_query_test/test/nikita4.png"
    dists = []
    for i, test in enumerate(TEST_PATHS):
        print(f"----------------------Image {i + 1}-----------------------")
        print(test)
        engine_runner = SingleQueryModel(MODEL, WEIGHT_PATH, QUERY_PATH)
        dist = engine_runner.test_image(test)
        print(dist)
        dists.append(dist)
    #print(TEST_PATHS)
    #print(dists)
   # dists = np.array(dists)

    # THRESHOLD_DIST = 0.01  # HIGHLY suspect prechosen hyperparameter, but dunno how else to tell
    # indices = np.where(dists < THRESHOLD_DIST)[0]
    # print(indices)
    # print(dists[indices])
    # matching_paths = np.array(TEST_PATHS)[indices]
    # print(matching_paths)
    # with open('dist.pkl', 'wb') as handle:
    #     pickle.dump(dists, handle, protocol=pickle.HIGHEST_PROTOCOL)

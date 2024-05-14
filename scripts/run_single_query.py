import torchreid
from torchvision.io import read_image
import torch
import os
import pickle
import numpy as np

def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


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

class EngineRunner:
    def __init__(self, query_path):
        self.model = None
        self.query_path = query_path

        # Should only be used for testing.
        # In practice, replace with on the fly image to test from YOLO
        self.gallery_path = None

    """
    :param test_path: string path to new test image
    """
    def test_image(self, test_path):
        self.gallery_path = test_path
        return self.run()

    def initialize_model(self):
        MODEL_NAME = "osnet_x0_5"
        model = torchreid.models.build_model(
            name=MODEL_NAME,
            num_classes=2,  # number of training identities, so might not matter
            loss="softmax",
            pretrained=True
        )
        torchreid.utils.load_pretrained_weights(model, "pretrained/osnet_x0_5_market.pth")
        model.eval()  # don't want to train
        self.model = model

    """
    :param path: string path to image
    :return: Tensor[image_channels, image_height, image_width]
    """

    def load_image(self, path):
        return read_image(path)

    def extract_features(self, img):
        with torch.no_grad():
            return self.model(img.float())

    def run(self):
        self.initialize_model()
        # 1.) import query and gallery img as tensors
        query = self.load_image(self.query_path)
        gallery = self.load_image(self.gallery_path)

        # add batch dimension (N=1)
        query = query.unsqueeze(0)
        gallery = gallery.unsqueeze(0)

        # 2.) run model on tensor inputs
        query_features = self.extract_features(query)
        gallery_features = self.extract_features(gallery)

        # not sure what this does, but scared what happens if I don't
        query_features = query_features.cpu()
        gallery_features = gallery_features.cpu()
        # 3.) compute distance. Before, was distance matrix, but also had many queries and galleries, so unnecessary.
        # try euclidean distance
        dist = euclidean_squared_distance(query_features, gallery_features)
        dist = dist.numpy()
        return dist[0][0]
        # correct: 22522.7500
        # wrong: 70822.5000
        # 4.) Somehow use distance value and determine if counts as a match or not


if __name__ == "__main__":
    THRESHOLD_DIST = 25000  # HIGHLY suspect prechosen hyperparameter, but dunno how else to tell
    #directory = 'single_query_test/gallery'
    directory = 'reid-data/grid/underground_reid/probe'
    TEST_PATHS = get_all_file_paths(directory)
    print(TEST_PATHS)

    QUERY_PATH = "single_query_test/query/probe.jpeg"
   # TEST_PATHS = ["single_query_test/gallery/correct.jpeg", "single_query_test/gallery/wrong.jpeg", "single_query_test/gallery/wrong2.png"]
    dists = []
    for i,test in enumerate(TEST_PATHS):
        print(f"----------------------Image {i+1}-----------------------")
        engine_runner = EngineRunner(QUERY_PATH)
        dists.append(engine_runner.test_image(test))
    print(dists)
    dists = np.array(dists)
    with open('dist.pkl', 'wb') as handle:
        pickle.dump(dists, handle, protocol=pickle.HIGHEST_PROTOCOL)
    indices = np.where(dists < THRESHOLD_DIST)[0]
    print(indices)
    print(dists[indices])
    matching_paths = np.array(TEST_PATHS)[indices]
    print(matching_paths)

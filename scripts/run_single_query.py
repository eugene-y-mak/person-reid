import torchreid
from torchvision.io import read_image


class EngineRunner:
    def __init__(self, query_path, gallery_path):
        self.model = None
        self.query_path = query_path

        # Should only be used for testing.
        # In practice, replace with on the fly image to test from YOLO
        self.gallery_path = gallery_path

    def initialize_model(self):
        MODEL_NAME = "osnet_x0_5"
        model = torchreid.models.build_model(
            name=MODEL_NAME,
            num_classes=2,  # number of training identities, so might not matter
            loss="softmax",
            pretrained=True
        )
        torchreid.utils.load_pretrained_weights(model, "pretrained/osnet_x0_5_market.pth")
        self.model = model

    """
    :param path: string path to image
    :return: Tensor[image_channels, image_height, image_width]
    """

    def load_image(self, path):
        return read_image(path)

    def extract_features(self, img):
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

        # 3.) compute distance. Before, was distance matrix, but also had many queries and galleries, so unnecessary.
        print(query_features)
        # 4.) Somehow use distance value and determine if counts as a match or not



if __name__ == "__main__":
    engine_runner = EngineRunner("single_query_test/query/probe.jpeg", "single_query_test/gallery/correct.jpeg")
    engine_runner.run()
from scripts import graph_based_image_segmentation as sgs
from scripts import utils
from pathlib import Path


class TestGraphBasedImageSegmentation:

    def setup(self):
        path: Path = Path("../test_directory/test_images/clothes.jpg")
        self.image = utils.read_image(path)

    def test_something(self):
        self.setup()
        assert self.image.size > 0

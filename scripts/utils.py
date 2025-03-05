from pathlib import Path
import numpy as np
import cv2 as cv
import logging.config


def setup_logging(debug_filename: Path, verbose: bool = False):
    handlers: list = ["console", "file"] if verbose else []
    log_config: dict = {
        "version": 1,
        "formatters": {
            "short": {
                "format": "%(funcName)s | %(levelname)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": logging.INFO,
                "formatter": "short",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": logging.DEBUG,
                "formatter": "short",
                "filename": str(debug_filename),
                "mode": "w",
            },
        },
        "loggers": {
            "": {
                "handlers": handlers,
                "level": "DEBUG",
            },
        },
    }
    logging.config.dictConfig(log_config)


def scale_image(
    image: np.ndarray, scale: float, interpolation: int = cv.INTER_LINEAR
) -> np.ndarray:
    logger: logging.Logger = logging.getLogger(__name__)
    try:
        (h, w) = image.shape[:2]
        new_height, new_width = int(h * scale), int(w * scale)
        resized_image = cv.resize(
            image, (new_width, new_height), interpolation=interpolation
        )
        logger.debug(f"image scaled by {scale}, new shape: {resized_image.shape}")
    except (cv.error, ValueError, Exception) as e:
        raise RuntimeError("failed to scale image") from e
    return resized_image if resized_image.size > 0 else image


def gaussian_blur(image: np.ndarray, sigma: float = 0.8, k: int = 3) -> np.ndarray:
    logger: logging.Logger = logging.getLogger(__name__)
    try:
        result: np.ndarray = cv.GaussianBlur(src=image, ksize=[k, k], sigmaX=sigma)
        logger.debug(f"applied gaussian blur with sigma: {sigma}")
    except (cv.error, Exception) as e:
        raise RuntimeError("failed to apply gaussian blur") from e
    return result


def median_blur(image: np.ndarray, k: int = 3) -> np.ndarray:
    logger: logging.Logger = logging.getLogger(__name__)
    try:
        result: np.ndarray = cv.medianBlur(src=image, ksize=k)
        logger.debug(f"applied median blur with ksize: {k}")
    except (cv.error, Exception) as e:
        raise RuntimeError("failed to apply median blur") from e
    return result


def show_image(image: np.ndarray, name: str = "image") -> None:
    logger: logging.Logger = logging.getLogger(__name__)
    logger.debug(f"showing image: {name}")
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyWindow(name)


def read_image(image_path: Path) -> np.ndarray:
    logger: logging.Logger = logging.getLogger(__name__)
    logger.debug(f"reading image: {image_path}")
    try:
        image: np.ndarray = cv.imread(str(image_path))
        logger.debug(f"image shape: {image.shape}, {image.dtype}")
    except (FileNotFoundError, cv.error, Exception) as e:
        raise RuntimeError("failure during image read") from e
    return image


GRAYSCALE_TO_RGB_MAP = [
    (0, 0, 0),
    (127, 1, 31),
    (254, 2, 62),
    (125, 3, 93),
    (252, 4, 124),
    (123, 5, 155),
    (250, 6, 186),
    (121, 7, 217),
    (248, 8, 248),
    (119, 9, 23),
    (246, 10, 54),
    (117, 11, 85),
    (244, 12, 116),
    (115, 13, 147),
    (242, 14, 178),
    (113, 15, 209),
    (240, 16, 240),
    (111, 17, 15),
    (238, 18, 46),
    (109, 19, 77),
    (236, 20, 108),
    (107, 21, 139),
    (234, 22, 170),
    (105, 23, 201),
    (232, 24, 232),
    (103, 25, 7),
    (230, 26, 38),
    (101, 27, 69),
    (228, 28, 100),
    (99, 29, 131),
    (226, 30, 162),
    (97, 31, 193),
    (224, 32, 224),
    (95, 33, 255),
    (222, 34, 30),
    (93, 35, 61),
    (220, 36, 92),
    (91, 37, 123),
    (218, 38, 154),
    (89, 39, 185),
    (216, 40, 216),
    (87, 41, 247),
    (214, 42, 22),
    (85, 43, 53),
    (212, 44, 84),
    (83, 45, 115),
    (210, 46, 146),
    (81, 47, 177),
    (208, 48, 208),
    (79, 49, 239),
    (206, 50, 14),
    (77, 51, 45),
    (204, 52, 76),
    (75, 53, 107),
    (202, 54, 138),
    (73, 55, 169),
    (200, 56, 200),
    (71, 57, 231),
    (198, 58, 6),
    (69, 59, 37),
    (196, 60, 68),
    (67, 61, 99),
    (194, 62, 130),
    (65, 63, 161),
    (192, 64, 192),
    (63, 65, 223),
    (190, 66, 254),
    (61, 67, 29),
    (188, 68, 60),
    (59, 69, 91),
    (186, 70, 122),
    (57, 71, 153),
    (184, 72, 184),
    (55, 73, 215),
    (182, 74, 246),
    (53, 75, 21),
    (180, 76, 52),
    (51, 77, 83),
    (178, 78, 114),
    (49, 79, 145),
    (176, 80, 176),
    (47, 81, 207),
    (174, 82, 238),
    (45, 83, 13),
    (172, 84, 44),
    (43, 85, 75),
    (170, 86, 106),
    (41, 87, 137),
    (168, 88, 168),
    (39, 89, 199),
    (166, 90, 230),
    (37, 91, 5),
    (164, 92, 36),
    (35, 93, 67),
    (162, 94, 98),
    (33, 95, 129),
    (160, 96, 160),
    (31, 97, 191),
    (158, 98, 222),
    (29, 99, 253),
    (156, 100, 28),
    (27, 101, 59),
    (154, 102, 90),
    (25, 103, 121),
    (152, 104, 152),
    (23, 105, 183),
    (150, 106, 214),
    (21, 107, 245),
    (148, 108, 20),
    (19, 109, 51),
    (146, 110, 82),
    (17, 111, 113),
    (144, 112, 144),
    (15, 113, 175),
    (142, 114, 206),
    (13, 115, 237),
    (140, 116, 12),
    (11, 117, 43),
    (138, 118, 74),
    (9, 119, 105),
    (136, 120, 136),
    (7, 121, 167),
    (134, 122, 198),
    (5, 123, 229),
    (132, 124, 4),
    (3, 125, 35),
    (130, 126, 66),
    (1, 127, 97),
    (128, 128, 128),
    (255, 129, 159),
    (126, 130, 190),
    (253, 131, 221),
    (124, 132, 252),
    (251, 133, 27),
    (122, 134, 58),
    (249, 135, 89),
    (120, 136, 120),
    (247, 137, 151),
    (118, 138, 182),
    (245, 139, 213),
    (116, 140, 244),
    (243, 141, 19),
    (114, 142, 50),
    (241, 143, 81),
    (112, 144, 112),
    (239, 145, 143),
    (110, 146, 174),
    (237, 147, 205),
    (108, 148, 236),
    (235, 149, 11),
    (106, 150, 42),
    (233, 151, 73),
    (104, 152, 104),
    (231, 153, 135),
    (102, 154, 166),
    (229, 155, 197),
    (100, 156, 228),
    (227, 157, 3),
    (98, 158, 34),
    (225, 159, 65),
    (96, 160, 96),
    (223, 161, 127),
    (94, 162, 158),
    (221, 163, 189),
    (92, 164, 220),
    (219, 165, 251),
    (90, 166, 26),
    (217, 167, 57),
    (88, 168, 88),
    (215, 169, 119),
    (86, 170, 150),
    (213, 171, 181),
    (84, 172, 212),
    (211, 173, 243),
    (82, 174, 18),
    (209, 175, 49),
    (80, 176, 80),
    (207, 177, 111),
    (78, 178, 142),
    (205, 179, 173),
    (76, 180, 204),
    (203, 181, 235),
    (74, 182, 10),
    (201, 183, 41),
    (72, 184, 72),
    (199, 185, 103),
    (70, 186, 134),
    (197, 187, 165),
    (68, 188, 196),
    (195, 189, 227),
    (66, 190, 2),
    (193, 191, 33),
    (64, 192, 64),
    (191, 193, 95),
    (62, 194, 126),
    (189, 195, 157),
    (60, 196, 188),
    (187, 197, 219),
    (58, 198, 250),
    (185, 199, 25),
    (56, 200, 56),
    (183, 201, 87),
    (54, 202, 118),
    (181, 203, 149),
    (52, 204, 180),
    (179, 205, 211),
    (50, 206, 242),
    (177, 207, 17),
    (48, 208, 48),
    (175, 209, 79),
    (46, 210, 110),
    (173, 211, 141),
    (44, 212, 172),
    (171, 213, 203),
    (42, 214, 234),
    (169, 215, 9),
    (40, 216, 40),
    (167, 217, 71),
    (38, 218, 102),
    (165, 219, 133),
    (36, 220, 164),
    (163, 221, 195),
    (34, 222, 226),
    (161, 223, 1),
    (32, 224, 32),
    (159, 225, 63),
    (30, 226, 94),
    (157, 227, 125),
    (28, 228, 156),
    (155, 229, 187),
    (26, 230, 218),
    (153, 231, 249),
    (24, 232, 24),
    (151, 233, 55),
    (22, 234, 86),
    (149, 235, 117),
    (20, 236, 148),
    (147, 237, 179),
    (18, 238, 210),
    (145, 239, 241),
    (16, 240, 16),
    (143, 241, 47),
    (14, 242, 78),
    (141, 243, 109),
    (12, 244, 140),
    (139, 245, 171),
    (10, 246, 202),
    (137, 247, 233),
    (8, 248, 8),
    (135, 249, 39),
    (6, 250, 70),
    (133, 251, 101),
    (4, 252, 132),
    (131, 253, 163),
    (2, 254, 194),
    (129, 255, 225),
]

if __name__ == "__main__":
    print("utility functions for image segmentation")
    print(f"using: \n open-cv:{cv.__version__}\n numpy {np.__version__}")

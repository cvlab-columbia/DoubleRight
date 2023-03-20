from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import PIL.Image
from PIL import ImageFile

from .utils import download_and_extract_archive
from .vision import VisionDataset
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SUN397(VisionDataset):
    """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.

    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of
    397 categories with 108'754 images.

    Args:
        root (string): Root directory of the dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _DATASET_URL = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    _DATASET_MD5 = "8ca2778205c41d23104230ba66911c7a"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        split=None,
        test_size=0.3,
        return_path: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._data_dir = Path(self.root) / "SUN397"

        if download:
            self._download()

        self.return_path = return_path

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        with open(self._data_dir / "ClassName.txt") as f:
            self.classes = [c[3:].strip() for c in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._image_files = list(self._data_dir.rglob("sun_*.jpg"))

        self._labels = [
            self.class_to_idx["/".join(path.relative_to(self._data_dir).parts[1:-1])] for path in self._image_files
        ]

        if split is not None:
            files_train, files_test, y_train, y_test = train_test_split(
                self._image_files, self._labels, test_size=test_size, random_state=42, shuffle=True
            )
            self._labels = y_train if split == 'train' else y_test
            self._image_files = files_train if split == 'train' else files_test

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        
        if self.return_path:
            return image, label, image_file
        else:
            return image, label

    def _check_exists(self) -> bool:
        return self._data_dir.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._DATASET_URL, download_root=self.root, md5=self._DATASET_MD5)

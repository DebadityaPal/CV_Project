from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os


class GANDataset(Dataset):
    def __init__(self, image_paths, label_paths):
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return self.image_paths[index], self.label_paths[index]


def test_file_name(image):
    ext = image.split('.')[-1]
    image = image.split('.')[0]
    image = image.split('_')[:-1]
    image = '_'.join(image)
    image = image + '.' + ext
    image = image.replace('INPUT_IMAGES', 'GT_IMAGES')
    return image


def create_train_test_eval_sets(exposure_errors_path=None, lol_path=None, uieb_path=None):

    exposure_errors_path = exposure_errors_path
    lol_path = lol_path
    uieb_path = uieb_path

    train_dataset = []
    test_dataset = []
    val_dataset = []

    train_labels = []
    test_labels = []
    val_labels = []

    if exposure_errors_path is not None:
        train_image_paths = [
            os.path.join(exposure_errors_path,
                         'training', 'INPUT_IMAGES', x)
            for x in os.listdir(os.path.join(exposure_errors_path, 'training', 'INPUT_IMAGES'))
        ]
        test_image_paths = [
            os.path.join(exposure_errors_path,
                         'testing', 'INPUT_IMAGES', x)
            for x in os.listdir(os.path.join(exposure_errors_path, 'testing', 'INPUT_IMAGES'))
        ]
        val_image_paths = [
            os.path.join(exposure_errors_path,
                         'validation', 'INPUT_IMAGES', x)
            for x in os.listdir(os.path.join(exposure_errors_path, 'validation', 'INPUT_IMAGES'))
        ]

        train_label_paths = [test_file_name(
            train_image_path) for train_image_path in train_image_paths]
        test_label_paths = [test_file_name(
            test_image_path) for test_image_path in test_image_paths]
        val_label_paths = [test_file_name(val_image_path)
                           for val_image_path in val_image_paths]

        train_dataset.extend(train_image_paths)
        test_dataset.extend(test_image_paths)
        val_dataset.extend(val_image_paths)
        train_labels.extend(train_label_paths)
        test_labels.extend(test_label_paths)
        val_labels.extend(val_label_paths)

    if lol_path is not None:
        train_image_paths = [
            os.path.join(lol_path, 'our485', 'low', x)
            for x in os.listdir(os.path.join(lol_path, 'our485', 'low'))
        ]
        train_label_paths = [
            os.path.join(lol_path, 'our485', 'high', x)
            for x in os.listdir(os.path.join(lol_path, 'our485', 'high'))
        ]
        train_image_paths += [
            os.path.join(lol_path, 'BrighteningTrain', 'low', x)
            for x in os.listdir(os.path.join(lol_path, 'BrighteningTrain', 'low'))
        ]
        train_label_paths += [
            os.path.join(lol_path, 'BrighteningTrain', 'high', x)
            for x in os.listdir(os.path.join(lol_path, 'BrighteningTrain', 'high'))
        ]
        eval_image_paths = [
            os.path.join(lol_path, 'eval15', 'low', x)
            for x in os.listdir(os.path.join(lol_path, 'eval15', 'low'))
        ]
        eval_label_paths = [
            os.path.join(lol_path, 'eval15', 'high', x)
            for x in os.listdir(os.path.join(lol_path, 'eval15', 'high'))
        ]
        train_image_paths, test_image_paths = train_test_split(
            train_image_paths, test_size=0.2, random_state=42)
        train_label_paths, test_label_paths = train_test_split(
            train_label_paths, test_size=0.2, random_state=42)

        train_dataset.extend(train_image_paths)
        test_dataset.extend(test_image_paths)
        val_dataset.extend(eval_image_paths)
        train_labels.extend(train_label_paths)
        test_labels.extend(test_label_paths)
        val_labels.extend(eval_label_paths)

    if uieb_path is not None:
        train_image_paths = [
            os.path.join(uieb_path, 'raw-890', x)
            for x in os.listdir(os.path.join(uieb_path, 'raw-890'))
        ]
        train_label_paths = [
            os.path.join(uieb_path, 'reference-890', x)
            for x in os.listdir(os.path.join(uieb_path, 'reference-890'))
        ]
        train_image_paths, test_image_paths = train_test_split(
            train_image_paths, test_size=0.2, random_state=42)
        train_label_paths, test_label_paths = train_test_split(
            train_label_paths, test_size=0.2, random_state=42)
        train_image_paths, eval_image_paths = train_test_split(
            train_image_paths, test_size=0.2, random_state=42)
        train_label_paths, eval_label_paths = train_test_split(
            train_label_paths, test_size=0.2, random_state=42)

        train_dataset.extend(train_image_paths)
        test_dataset.extend(test_image_paths)
        val_dataset.extend(eval_image_paths)
        train_labels.extend(train_label_paths)
        test_labels.extend(test_label_paths)
        val_labels.extend(eval_label_paths)

    return train_dataset, test_dataset, val_dataset, train_labels, test_labels, val_labels

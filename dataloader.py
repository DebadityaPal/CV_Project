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


def test_file_name(image_filepaths):
    res = []
    for image in image_filepaths:
        ext = image.split('.')[-1]
        image = image.split('.')[0]
        image = image.split('_')[:-1]
        image = '_'.join(image)
        image = image + '.' + ext
        image.replace('INPUT_IMAGES', 'GT_IMAGES')
        res.append(image)
    return res


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
        print("Loading exposure errors dataset")
        # list of images
        train_set = [
            os.path.join(exposure_errors_path,
                         'training', 'INPUT_IMAGES', x)
            for x in os.listdir(os.path.join(exposure_errors_path, 'training', 'INPUT_IMAGES'))
        ]
        test_set = [
            os.path.join(exposure_errors_path,
                         'testing', 'INPUT_IMAGES', x)
            for x in os.listdir(os.path.join(exposure_errors_path, 'testing', 'INPUT_IMAGES'))
        ]
        val_set = [
            os.path.join(exposure_errors_path,
                         'validation', 'INPUT_IMAGES', x)
            for x in os.listdir(os.path.join(exposure_errors_path, 'validation', 'INPUT_IMAGES'))
        ]

        train_dataset += train_set
        test_dataset += test_set
        val_dataset += val_set

        # list of labels
        train_labels += test_file_name(train_set)
        test_labels += test_file_name(test_set)
        val_labels += test_file_name(val_set)

    if lol_path is not None:
        print("Loading LOL dataset")
        # list of image names
        total_dataset = []
        total_labels = []
        total_dataset += [
            os.path.join(lol_path, 'our485', 'low', x)
            for x in os.listdir(os.path.join(lol_path, 'our485', 'low'))
        ]
        total_dataset += [
            os.path.join(lol_path, 'BrighteningTrain', 'low', x)
            for x in os.listdir(os.path.join(lol_path, 'BrighteningTrain', 'low'))
        ]
        total_labels += [
            os.path.join(lol_path, 'our485', 'high', x)
            for x in os.listdir(os.path.join(lol_path, 'our485', 'high'))
        ]
        total_labels += [
            os.path.join(lol_path, 'BrighteningTrain', 'high', x)
            for x in os.listdir(os.path.join(lol_path, 'BrighteningTrain', 'high'))
        ]
        train_set, test_set = train_test_split(
            total_dataset, test_size=0.2, random_state=42)
        train_labels, test_labels = train_test_split(
            total_labels, test_size=0.2, random_state=42)
        train_dataset += train_set
        test_dataset += test_set
        train_labels += train_labels
        test_labels += test_labels

        val_dataset += [
            os.path.join(lol_path, 'eval15', 'low', x)
            for x in os.listdir(os.path.join(lol_path, 'eval15', 'low'))
        ]
        val_labels += [
            os.path.join(lol_path, 'eval15', 'high', x)
            for x in os.listdir(os.path.join(lol_path, 'eval15', 'high'))
        ]

    if uieb_path is not None:
        print("Loading UIEB dataset")
        total_dataset = []
        total_labels = []
        total_dataset += [
            os.path.join(uieb_path, 'raw-890', x)
            for x in os.listdir(os.path.join(uieb_path, 'raw-890'))
        ]
        total_labels += [
            os.path.join(uieb_path, 'reference-890', x)
            for x in os.listdir(os.path.join(uieb_path, 'reference-890'))
        ]
        train_set, test_set = train_test_split(
            total_dataset, test_size=0.2, random_state=42)
        train_labels, test_labels = train_test_split(
            total_labels, test_size=0.2, random_state=42)
        train_set, val_set = train_test_split(
            train_set, test_size=0.1, random_state=42)
        train_labels, val_labels = train_test_split(
            train_labels, test_size=0.1, random_state=42)

        train_dataset += train_set
        test_dataset += test_set
        val_dataset += val_set
        train_labels += train_labels
        test_labels += test_labels
        val_labels += val_labels

    return train_dataset, test_dataset, val_dataset, train_labels, test_labels, val_labels

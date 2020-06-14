import json
import os
import torch
import argparse
from DeepNetworks.MISRGRU import MISRGRU
from DataLoader import ImagesetDataset
from utils import generate_submission_file, get_imageset_directories


def test(config):
    checkpoint_path = os.path.join(config["paths"]["checkpoint_dir"], 'MISR-GRU.pth')
    assert os.path.isfile(checkpoint_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MISRGRU(config["network"]).to(device)
    model_dict = torch.load(checkpoint_path)
    model.load_state_dict(model_dict)

    config["training"]["create_patches"] = False
    data_directory = config["paths"]["data_dir"]
    test_set_directories = get_imageset_directories(os.path.join(data_directory, "test"))
    test_dataset = ImagesetDataset(imset_dir=test_set_directories, config=config["training"], top_k=-1)

    generate_submission_file(model, imset_dataset=test_dataset, out=config["paths"]["submission_dir"])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", help="path of the config file", default='../config.json')

    args = parser.parse_args()
    assert os.path.isfile(args.config_file_path)

    with open(args.config_file_path, "r") as read_file:
        config = json.load(read_file)

    test(config)

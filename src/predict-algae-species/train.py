import os
import json
import time
from tqdm import tqdm
from imutils import paths
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from core.dataloader import SegmentationDataSet
from core.model import UNet
from core.callbacks import EarlyStopping

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

__DEVICE__ = "cuda" if torch.cuda.is_available() else "cpu"
__PIN_MEMORY__ = True if __DEVICE__ == "cuda" else False
config = {
    "ROOT": "../../tests/resources",
    "IMAGE_DATASET_PATH": "tiles",
    "MASK_DATASET_PATH": "annotations",
    "ALGAE_SPECIES": ["Pp", "Cr", "Cv"],
    "TEST_SPLIT": 0.2,
    "NUM_CHANNELS": 1,
    "NUM_CLASSES": 1,
    "NUM_LEVELS": 3,
    "INIT_LR": 0.001,
    "NUM_EPOCHS": 40,
    "BATCH_SIZE": 16,
    "INPUT_IMAGE_WIDTH": 512,
    "INPUT_IMAGE_HEIGHT": 512,
    "THRESHOLD": 0.5,
    "PATIENCE": 5,
    "BASE_OUTPUT": "output",
    "MODEL_PATH": "unet.ptm",
    "TEST_PATHS": "test_paths.txt"
}


def get_dataset(strain: str, raw: list, annotation: list):
    raw = sorted([x for x in raw if strain in x])
    annotation = sorted([x for x in annotation if strain in x])
    return raw, annotation


def main():
    base_output = os.path.join(config["ROOT"], config["BASE_OUTPUT"])
    if not os.path.exists(base_output):
        os.mkdir(base_output)
    model_path = os.path.join(base_output, config["MODEL_PATH"])
    test_path = os.path.sep.join([base_output, config["TEST_PATHS"]])

    # sort and then load raw and annotation images
    raw_paths = os.path.join(config["ROOT"], config["IMAGE_DATASET_PATH"])
    annotation_paths = os.path.join(config["ROOT"], config["MASK_DATASET_PATH"])
    raw_paths = sorted(list(paths.list_images(raw_paths)))
    annotation_paths = sorted(list(paths.list_images(annotation_paths)))

    # partition the data into training and testing splits
    split = train_test_split(raw_paths,
                             annotation_paths,
                             test_size=config["TEST_SPLIT"],
                             random_state=42)

    (train_images, test_images) = split[:2]
    (train_annotations, test_annotations) = split[2:]

    # write the test image paths to disk so that we can use them when evaluating/testing our model later
    print("[INFO] saving testing image paths...")
    with open(test_path, "w") as f:
        f.write("\n".join(test_images))

    # define some transformations to later use when we load the dataset
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((config["INPUT_IMAGE_HEIGHT"],
                                                       config["INPUT_IMAGE_WIDTH"])),
                                    transforms.ToTensor()])

    with open('metrics.txt', 'w') as outfile:
        outfile.write(f'\t Loss output for each algae model:')
    # loop through each strain and create a model and save it into our `output` folder
    for strain in config["ALGAE_SPECIES"]:
        train_image_subset, train_annotation_subset = get_dataset(strain, train_images, train_annotations)
        # create the train dataset
        train_ds = SegmentationDataSet(image_paths=train_image_subset,
                                       mask_paths=train_annotation_subset,
                                       transform=transform)
        test_image_subset, test_annotation_subset = get_dataset(strain, test_images, test_annotations)
        # create the test dataset
        test_ds = SegmentationDataSet(image_paths=test_image_subset,
                                      mask_paths=test_annotation_subset,
                                      transform=transform)

        print(f"[INFO] found {len(train_ds)} samples in the training set...")
        print(f"[INFO] found {len(test_ds)} samples in the test set...")

        # create the training data loaders
        train_loader = DataLoader(train_ds,
                                  shuffle=True,
                                  batch_size=config["BATCH_SIZE"],
                                  pin_memory=__PIN_MEMORY__,
                                  num_workers=os.cpu_count())
        # create the test data loaders
        test_loader = DataLoader(test_ds, shuffle=False,
                                 batch_size=config["BATCH_SIZE"],
                                 pin_memory=__PIN_MEMORY__,
                                 num_workers=os.cpu_count())

        # initialize our UNet model
        unet = UNet(out_size=(config["INPUT_IMAGE_HEIGHT"], config["INPUT_IMAGE_WIDTH"])).to(__DEVICE__)

        lossFunc = BCEWithLogitsLoss()  # initialize loss function
        optimisation = Adam(unet.parameters(), lr=config["INIT_LR"])  # initialize optimiser

        # calculate steps per epoch for training and test set
        train_steps = len(train_ds) // config["BATCH_SIZE"]
        test_steps = len(test_ds) // config["BATCH_SIZE"]

        history = {"train_loss": [], "test_loss": []}  # initialize a dictionary to store training history

        # initialize the early_stopping object

        early_stopping = EarlyStopping(patience=config["PATIENCE"],
                                       verbose=True,
                                       path=model_path + '.{}'.format(strain))

        print("[INFO] training the network for {} algae strain...".format(strain))
        start_time = time.time()
        for e in tqdm(range(config["NUM_EPOCHS"])):
            # set the model in training mode
            unet.train()
            # initialize the total training and validation loss
            total_train_loss = 0
            total_test_loss = 0

            # loop over the training set
            for (i, (x, y)) in enumerate(train_loader):
                (x, y) = (x.to(__DEVICE__), y.to(__DEVICE__))
                # perform a forward pass and calculate the training loss
                pred = unet(x)
                loss = lossFunc(pred, y)
                optimisation.zero_grad()  # reset gradient
                loss.backward()  # compute backprop
                optimisation.step()  # update model parameters
                total_train_loss += loss  # add the loss to the total training loss

            # switch off autograd
            with torch.no_grad():
                unet.eval()  # set the model in evaluation mode
                for (x, y) in test_loader:
                    (x, y) = (x.to(__DEVICE__), y.to(__DEVICE__))
                    pred = unet(x)  # make predictions
                    total_test_loss += lossFunc(pred, y)  # get validation loss

            # calculate the average training and validation loss
            avg_train_loss = total_train_loss / train_steps
            avg_test_loss = total_test_loss / test_steps

            # update our training history
            history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
            history["test_loss"].append(avg_test_loss.cpu().detach().numpy())

            # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, config["NUM_EPOCHS"]))
            print("Train loss: {:.6f}, Test loss: {:.4f}".format(avg_train_loss, avg_test_loss))

            # see if early stopping gets triggered
            early_stopping(avg_test_loss, unet)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        with open('metrics.txt', 'a') as outfile:
            outfile.write(f'\n Mean training loss for {strain} {avg_train_loss}')

        end_time = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))

        """ Once training is done, we can plot the loss for each strain and see how our model performed.
        We also save each model to use later for inference """
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history["train_loss"], label="train_loss")
        plt.plot(history["test_loss"], label="test_loss")
        plt.title("Training Loss for {}".format(strain))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(f'{strain}_model_results.png', dpi=120)


if __name__ == '__main__':
    main()

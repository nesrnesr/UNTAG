import argparse
import os
import pathlib
import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    classification_report,
    roc_curve,
)
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ForgeryNet
from model import RegionSplicerNet

"""Stage 2"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="path of stage 1 results e.g ./tb_logs")
    parser.add_argument(
        "--num_classes",
        default=6,
        help="Number of transformations applied by R-splicer",
    )
    parser.add_argument("--exp_name", help="experiment's name")
    parser.add_argument("--data", help="path to dataset root.")
    parser.add_argument("--df_type", default="cdf")
    parser.add_argument("--batch_size", default=512)
    parser.add_argument("--fitted_gde", type=str, help="load prefitted gde")
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument(
        "--save_exp",
        default=pathlib.Path(__file__).parent / "detector_exp",
        help="Save fitted models and roc curves",
    )
    args = parser.parse_args()
    return args


class DeepfakeDetector:
    def __init__(
        self,
        weights,
        batch_size,
        device="cuda",
        deepfake_type="cdf",
        fitted_gde=None,
        num_classes=6,
        encoder="resnet18",
    ):
        """
        Deepfake Detector

        args:
        weights[str] - path to stage 1 weights
        device[str] - device on wich model should be run
        deepfake_type[str] - name of the forgery to testing
        fitted_gde[str] - path of an available gde model
        num_classes[int] - number of transformations applied by R-Splicer

        """
        self.spliceregion_model = self.model(device, weights, num_classes, encoder)
        self.batch_size = batch_size
        self.deepfake_type = deepfake_type
        self.fitted_gde = fitted_gde
        self.device = device
        self.auc_results = {}

    @staticmethod
    def model(device, weights, num_classes, encoder):
        model = RegionSplicerNet(
            pretrained=False, num_class=num_classes, encoder=encoder
        )
        state_dict = torch.load(weights)["state_dict"]
        state_dict = {i.replace("model.", ""): j for i, j in state_dict.items()}
        model.load_state_dict(state_dict)
        print("loaded model state!")

        model.to(device)
        model.eval()
        return model

    @staticmethod
    def roc_auc(labels, scores, forgery_name=None, save_path=None, draw_graph=False):
        fpr, tpr, thresh = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        if draw_graph:
            plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
            plt.legend(loc="lower right")
            plt.plot([0, 1], [0, 1], "r--")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")

            save_images = save_path if save_path else "./roc_results"
            os.makedirs(save_images, exist_ok=True)
            image_path = (
                os.path.join(save_images, forgery_name + "_roc.png")
                if forgery_name
                else os.path.join(save_images, "roc_curve.png")
            )
            plt.savefig(image_path, dpi=300)
            plt.close()
        return roc_auc

    def create_test_embeds(self, path_to_images):
        """Extract embeddings using stage 1"""
        embeddings = []
        labels = []
        images = []
        dataset = ForgeryNet(
            test_images=path_to_images, mode="test", deepfake_type=self.deepfake_type
        )
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, num_workers=16
        )
        with torch.no_grad():
            for imgs, lbls in tqdm(dataloader):
                _, _, embeds = self.spliceregion_model(imgs.to(self.device))
                embeddings.append(embeds.to("cpu"))
                labels.append(lbls.to("cpu"))
                images.append(imgs.to("cpu"))
                torch.cuda.empty_cache()

        return torch.cat(embeddings), torch.cat(labels), torch.cat(images)

    def create_train_embeds(self, path_to_images):
        """extrats embeddings of training data
        Args:
            path_to_images [str]: path of trainset

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: embeds, labels
        """
        embeddings = []
        labels = []

        dataset = ForgeryNet(train_images=path_to_images, mode="train", stage1=False)
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, num_workers=16
        )
        with torch.no_grad():
            print("Generating training embeddings...")
            for (i_batch, imgs) in enumerate(tqdm(dataloader)):
                n = (
                    self.batch_size
                    if (i_batch <= len(dataset) / self.batch_size - 1)
                    else len(dataset) % self.batch_size
                )
                (_, lbls) = torch.meshgrid(
                    torch.arange(0, n), torch.arange(0, len(imgs)), indexing="xy"
                )
                imgs = torch.concat(imgs)
                lbls = lbls.to(dtype=torch.int).flatten()
                _, _, embeds = self.spliceregion_model(imgs.to(self.device))
                assert len(embeds) == len(lbls) == len(imgs), IndexError(
                    f"Shape mismatch: len(embeds), len(lbls), (imgs): {len(embeds), len(lbls), len(imgs)}"
                )
                embeddings.append(embeds.to("cpu"))
                labels.append(lbls.to("cpu"))
                torch.cuda.empty_cache()
        return torch.cat(embeddings), torch.cat(labels)

    @staticmethod
    def GDE_fit(train_embeds, save_path=None, num_components=3):
        """Fits a Gaussian Mixture Model"""

        train_embeds = torch.from_numpy(normalize(train_embeds))

        gde = GMM(
            n_components=num_components,
            verbose=1,
            verbose_interval=5,
            init_params="kmeans",
            max_iter=250,
        ).fit(train_embeds)

        print("finished GDE fitting")
        if save_path:
            filename = os.path.join(save_path, f"gde_{num_components}.sav")
            pickle.dump(gde, open(filename, "wb"))
        return gde

    @staticmethod
    def GDE_scores(embeds, gde):
        embeds = torch.from_numpy(normalize(embeds))
        scores = -gde.score_samples(embeds)
        return scores

    def GDE_pipeline(
        self,
        test_embeds,
        test_labels,
        train_embeds=None,
        save_path=None,
        images=None,
    ):
        if self.fitted_gde is not None:
            with open(self.fitted_gde, "rb") as pickle_file:
                GDE_model = pickle.load(pickle_file)
                print("Loaded a prefitted gde_model!")
        else:
            assert train_embeds is not None
            print("Fitting gde...")
            GDE_model = self.GDE_fit(train_embeds, save_path, num_components=3)

        gde_scores = self.GDE_scores(test_embeds, GDE_model)
        fpr, tpr, thresholds = roc_curve(test_labels, gde_scores, pos_label=1)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        predictions = [1 if score >= optimal_threshold else 0 for score in gde_scores]


        with open(os.path.join(str(save_path), "predictions.txt"), "w") as file:
            file.write("Predictions\n")
            file.write(" ".join(map(str, predictions)))
            file.write("\n\n\n")
            file.write("Labels\n")
            file.write(" ".join(map(str, test_labels.tolist())))
            file.write("\n\n\n")
            file.write("Balanced Accuracy\n")
            file.write(str(balanced_accuracy_score(test_labels, predictions)))
            file.write("\n\n\n")
            file.write(classification_report(test_labels, predictions))

        self.auc_results.update(
            {
                "GMM_AUC": self.roc_auc(
                    test_labels, gde_scores, "gde_3_" + self.deepfake_type, save_path
                ),
                "Threshold": optimal_threshold
            }
        )
        with open(os.path.join(str(save_path), "AUC_resuts.txt"), "a+") as f:
            f.write(f"{self.auc_results} \n")
            f.write(f"{balanced_accuracy_score(test_labels, predictions)}\n")

    def detect(self, dataset_root, save_path=None):
        """
        Runs full deepfake detection pipeline with generation of relevant visualizations.

        Args:
        dataset_root[str] - path to the data: train + test data.
        """

        train_images = os.path.join(dataset_root, "train")
        test_images = os.path.join(dataset_root, "test")

        if not self.fitted_gde:
            train_embeds, _ = self.create_train_embeds(train_images)
        else:
            train_embeds=None

        test_embeds, test_labels, images = self.create_test_embeds(test_images)
        print("generated test embeddings and their labels.")

        self.GDE_pipeline(
            test_embeds,
            test_labels,
            train_embeds=train_embeds,
            save_path=save_path,
            images=images,
        )


if __name__ == "__main__":

    sns.set(style="white")
    sns.set_palette(["#FF7518", "#090364"])

    args = get_args()

    checkpoint_path = pathlib.Path(args.checkpoint)

    epoch_name = f"{os.path.basename(checkpoint_path)[:-5]}_results"

    detector = DeepfakeDetector(
        weights=str(checkpoint_path),
        batch_size=args.batch_size,
        deepfake_type=args.df_type,
        encoder=args.encoder,
        num_classes=args.num_classes,
        fitted_gde=args.fitted_gde,
    )
    # setup the exp name folder
    save_path = os.path.join(args.save_exp, args.exp_name)
    os.makedirs(save_path, exist_ok=True)

    # setup the epoch name folder
    save_path = os.path.join(save_path, epoch_name)
    os.makedirs(save_path, exist_ok=True)

    detector.detect(args.data, save_path)

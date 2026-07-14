import pandas as pd
import json

class DifficultyScore:
    """
    Difficulty Score Module

    This module:
    1. Loads precomputed normalized lesion features.
    2. Computes the final difficulty score.
    3. Computes adaptive positive assignment budget (k).
    4. Computes adaptive diversity weight (lambda_div).
    """

    def __init__(
        self,
        csv_path,
        annotation_path,
        alpha=0.25,
        beta=0.40,
        gamma=0.35,
        k_min=4,
        k_max=20,
        lambda_min=0.10,
        lambda_max=1.00
    ):

        # Feature weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Adaptive supervision limits
        self.k_min = k_min
        self.k_max = k_max

        # Diversity loss limits
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # Load CSV
        self.df = pd.read_csv(csv_path)
        with open(annotation_path, "r") as f:
          coco = json.load(f)

        self.id_to_name = {}        #######################################################################

        for img in coco["images"]:
            self.id_to_name[img["id"]] = img["file_name"]

        # Compute everything once
        self.compute_difficulty()

    ########################################################

    def compute_difficulty(self):
        """
        Compute final difficulty score
        """

        self.df["difficulty_score"] = (

            self.alpha * self.df["size_norm"]

            + self.beta * self.df["border_norm"]

            + self.gamma * self.df["clarity_difficulty"]

        )

        # Keep values between 0 and 1
        self.df["difficulty_score"] = self.df["difficulty_score"].clip(0, 1)

        # Compute adaptive parameters
        self.df["adaptive_k"] = self.df["difficulty_score"].apply(
            self.compute_adaptive_k
        )

        self.df["lambda_div"] = self.df["difficulty_score"].apply(
            self.compute_lambda_div
        )

        # Dictionary for fast lookup
        self.lookup = self.df.set_index("image_name").to_dict("index")

    ########################################################

    def compute_adaptive_k(self, difficulty):

        """
        Adaptive Positive Assignment

        k = k_min + (k_max-k_min)*D
        """

        k = self.k_min + (self.k_max - self.k_min) * difficulty

        return int(round(k))

    ########################################################

    def compute_lambda_div(self, difficulty):

        """
        Adaptive Diversity Weight

        λ = λ_min + (λ_max-λ_min)*D
        """

        return (

            self.lambda_min

            + (self.lambda_max - self.lambda_min)

            * difficulty

        )

    ########################################################

    def get_difficulty(self, image_id):

        """
        Return all information for one image.
        """
        image_name = self.id_to_name[int(image_id)]

        if image_name not in self.lookup:

            raise ValueError(f"{image_name} not found.")

        return self.lookup[image_name]


############################################################
# Example
############################################################

if __name__ == "__main__":

    module = DifficultyScore("final_difficulty_scores.csv", "instances_train.json")

    info = module.get_difficulty(0)

    print(info)

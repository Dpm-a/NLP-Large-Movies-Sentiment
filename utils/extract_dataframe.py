import os
import pandas as pd
from tqdm import tqdm

PATH = "aclImdb/"

# create DF


def create_df(path, train_test, pos_neg):
    df = pd.DataFrame(columns=["review", "rating"])

    # Iterating trough file names ordered by number
    for file in tqdm(sorted(os.listdir(path + train_test + pos_neg), key=lambda x: int(x.split("_")[0])),
                     desc=f'Creating {train_test[:-1].capitalize()} {"Positive" if pos_neg == "pos/" else "Negative"}_Review file'):

        # Opening the file, retrieving review and rating
        with open(path + train_test + pos_neg + file, "r") as f:
            rating = file.split("_")[1].split(".")[0]
            txt = f.read()

            # writing line by line in a new df
            new_row = pd.DataFrame({'review': [txt], 'rating': [int(rating)]})
            df = pd.concat([df, new_row], ignore_index=True)

    return df


def main():
    # TRAIN
    train_pos = create_df(PATH, "train/", "pos/")
    train_neg = create_df(PATH, "train/", "neg/")
    train_df = pd.concat([train_pos, train_neg], ignore_index=True)
    train_df.to_csv("../data/train.csv", index=False)

    # TEST
    test_pos = create_df(PATH, "test/", "pos/")
    test_neg = create_df(PATH, "test/", "neg/")
    test_df = pd.concat([test_pos, test_neg], ignore_index=True)
    test_df.to_csv("../data/test.csv", index=False)


if __name__ == "__main__":
    main()

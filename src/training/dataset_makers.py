from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd

def make_dataset(dname):
  if dname == "sst-2":
    dataset = load_dataset("glue", "sst2")
  elif dname == "sst-5":
    dataset = load_dataset("SetFit/sst5")
  elif dname == "yelp-5":
    dataset = load_dataset("Yelp/yelp_review_full")
    train_subset = dataset["train"].select(range(50000))      # First 50K instances of training set
    test_subset = dataset["test"].select(range(10000))        # First 10K instances of test set

    # Combine the subsets into a single DatasetDict
    dataset = DatasetDict({
      "train": train_subset,
      "test": test_subset,
      "validation": test_subset
    })
  elif dname == "irony":
    dataset = prep_irony()
  elif dname == "offense":
    dataset = prep_offense()
  elif dname == "stance":
    dataset = prep_stance()
  return dataset

    
def prep_irony():
  train_csv_path = "../data/irony/train.csv"  # Replace with the path to your train.csv file
  test_csv_path = "../data/irony/test.csv"    # Replace with the path to your test.csv file

  train_df = pd.read_csv(train_csv_path, delimiter="\t")
  test_df = pd.read_csv(test_csv_path, delimiter="\t")

  # Verify the DataFrame structure
  print("Train DataFrame sample:")
  print(train_df.head())

  # Rename columns for consistency (optional)
  train_df.rename(columns={"Tweet text": "text", "Label": "label"}, inplace=True)
  test_df.rename(columns={"Tweet text": "text", "Label": "label"}, inplace=True)

  # Drop unnecessary columns (e.g., 'Tweet index') if needed
  train_df = train_df[["text", "label"]]
  test_df = test_df[["text", "label"]]

  # Convert pandas DataFrames to Hugging Face Datasets
  train_dataset = Dataset.from_pandas(train_df)
  test_dataset = Dataset.from_pandas(test_df)

  # Create a DatasetDict with train and test splits
  dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
    "validation": test_dataset,
  })
  return dataset


def prep_offense():
  train_csv_path = "../data/offense/train.csv"  # Replace with the path to your train.csv file

  train_df = pd.read_csv(train_csv_path, delimiter="\t")

  # Verify the DataFrame structure
  print("Train DataFrame sample:")
  print(train_df.head())

  # Rename columns for consistency (optional)
  train_df.rename(columns={"tweet": "text", "subtask_a": "label"}, inplace=True)


  tweets_df = pd.read_csv("../data/offense/test.csv", delimiter="\t")  # Assuming tab-separated file
  print("Tweets file preview:")
  print(tweets_df.head())

  # Load the second file (labels)
  labels_df = pd.read_csv("../data/offense/labels.csv", delimiter=",")  # Assuming no header
  print("\nLabels file preview:")
  print(labels_df.head())

  # Merge the two files on the 'id' column
  merged_df = pd.merge(tweets_df, labels_df, on="id", how="inner")
  merged_df.rename(columns={"tweet": "text"}, inplace=True)
  print("\nMerged file preview:")
  print(merged_df.head())

  # Drop unnecessary columns (e.g., 'Tweet index') if needed
  train_df = train_df[["text", "label"]]
  test_df = merged_df[["text", "label"]]
 
  # Convert pandas DataFrames to Hugging Face Datasets
  train_dataset = Dataset.from_pandas(train_df)
  test_dataset = Dataset.from_pandas(test_df)

  # Create a DatasetDict with train and test splits
  dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
    "validation": test_dataset
  })
  def map_labels(example):
    label_mapping = {"OFF": 1, "NOT": 0}
    example["label"] = label_mapping[example["label"]]
    return example
  dataset = dataset.map(map_labels)
  return dataset



def prep_stance():
  def load_stance(csv_path):
    """
    Load the STANCE16 dataset from CSV and preprocess it.
    Assumes the CSV has columns: text, target, stance, notes.
    """

    # Load CSV into a DataFrame
    df = pd.read_csv(csv_path, sep=',',lineterminator='\r',encoding = 'unicode_escape')
    print(df.head(10))

    # Map stance labels to integers
    label_map = {"FAVOR": 0, "AGAINST": 1, "NONE": 2}
    df["label"] = df["Stance"].map(label_map)

    # Remove unnecessary columns (like notes)
    df = df.drop(columns=["Sentiment", "Stance", "Opinion Towards"])


    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    return dataset

  train_data = load_stance("../data/stance/train.csv")
  test_data = load_stance("../data/stance/test.csv")
  dataset = DatasetDict({
    "train": train_data,
    "test": test_data,
    "validation": test_data
  })
  return dataset


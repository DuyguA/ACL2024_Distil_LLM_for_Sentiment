from huggingface_hub import hf_hub_download

teacher_model = "llama1B"

datasets = ["irony", "offense", "sst2", "sst5", "stance", "yelp5"]


for dataset in datasets:
  fname = f"embeddings-{teacher_model}-{dataset}.jsonl"
  hf_hub_download(repo_id="BayanDuygu/LlamaTensors", filename=fname)


for x in ["3", "8"]:
  dataset="sst5"
  fname = f"embeddings-llama{x}B-{dataset}.jsonl"
  hf_hub_download(repo_id="BayanDuygu/LlamaTensors", filename="data/train/" + fname, repo_type="dataset", local_dir=".")


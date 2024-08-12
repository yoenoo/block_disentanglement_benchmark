import os
import requests

def pull_imagenet_data_by_synset(synset: str, target_dir="./data"):
	url = f"https://image-net.org/data/winter21_whole/{synset}.tar"
	r = requests.get(url, stream=True)
	if r.status_code == 200:
		print("Processing...", r.url)
		target_path = os.path.join(target_dir, synset + ".tar")
		with open(target_path, "wb") as f:
			f.write(r.raw.read())

	print(f"Download complete: {synset}")


if __name__ == "__main__":
	# see below for the full list (imagenet-21k)
	# https://www.image-net.org/api/imagenet.attributes.obtain_synset_wordlist.php 
	synset = "n12102133" # grass
	pull_imagenet_data_by_synset(synset)

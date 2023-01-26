import urllib.request
from time import sleep

import pandas as pd

from pathlib import Path
from multiprocessing.pool import ThreadPool

unsplash_dataset_path = Path("instance/data/unsplash-lite")

# Read the photos table
photos = pd.read_csv(unsplash_dataset_path / "photos.tsv000", sep='\t', header=0)

# Extract the IDs and the URLs of the photos
photo_urls = photos[['photo_id', 'photo_image_url']].values.tolist()

# Print some statistics
print(f'Photos in the dataset: {len(photo_urls)}')

# Path where the photos will be downloaded
photos_donwload_path = unsplash_dataset_path / "photos"

# Function that downloads a single photo
def download_photo(photo):
    # Get the ID of the photo
    photo_id = photo[0]

    # Get the URL of the photo (setting the width to 640 pixels)
    photo_url = photo[1] + "?w=640"

    # Path where the photo will be stored
    photo_path = photos_donwload_path / (photo_id + ".jpg")

    # Only download a photo if it doesn't exist
    if not photo_path.exists():
        try:
            urllib.request.urlretrieve(photo_url, photo_path)
        except:
            # Catch the exception if the download fails for some reason
            print(f"Cannot download {photo_url}")
            pass

# Create the thread pool
threads_count = 16
pool = ThreadPool(threads_count)

# Start the download
pool.map(download_photo, photo_urls)

# Print some statistics
print(f'Photos downloaded: {len(photos)}')

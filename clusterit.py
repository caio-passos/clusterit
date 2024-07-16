import os
from sklearn.cluster import KMeans
import cv2
from tqdm import tqdm
import itertools

def extract_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

def load_images(folder, batch_size):
    file_list = os.listdir(folder)
    for filename in itertools.batched(file_list, batch_size):
        for file in filename:
            img_path = os.path.join(folder, file)
            image = cv2.imread(img_path)
            if image is not None:
                features = extract_histogram(image)
                yield (file, features)


def perform_kmeans(images, folder, n_clusters, batch_size):
    filenames, features = zip(*images)
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_predict(features)

    for cluster in range(n_clusters):
        os.makedirs(f"cluster_{cluster}", exist_ok=True)
    with tqdm(total=len(filenames), desc=f"--Clusterizando imagens -- batch_size: {batch_size}") as pbar:
        for filename, cluster in zip(filenames, clusters):
            image = cv2.imread(os.path.join(folder, filename))
            cv2.imwrite(f"cluster_{cluster}/{filename}", image)
            pbar.update(1)


if __name__ == "__main__":
    batch_size = 1
    # put the absolute filepath of where your images are located
    folder = "/home/user/images"
    images_generator = load_images(folder, batch_size)
    total_images = sum(1 for _ in os.listdir(folder))

    perform_kmeans(list(images_generator), folder, n_clusters=9, batch_size=batch_size)

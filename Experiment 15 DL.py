import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load butterfly image
img = mpimg.imread("C:\\Users\\Reddy\\Desktop\\Deep Learning Lab\\butterfly_files")

# Reshape image into pixel array
pixels = img.reshape((-1, 3))

# Number of clusters
K = 3

# Randomly choose cluster centers
centers = pixels[np.random.choice(len(pixels), K, replace=False)]

# K-means clustering
for i in range(10):
    distances = np.sqrt(((pixels - centers[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)

    for k in range(K):
        centers[k] = pixels[labels == k].mean(axis=0)
segmented = centers[labels]
segmented = segmented.reshape(img.shape)

# Show images
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Butterfly Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(segmented.astype(np.uint8))
plt.title("Segmented Butterfly Image")
plt.axis("off")

plt.show()

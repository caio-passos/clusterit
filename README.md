This script separate images in directories by color simmilarities. Each image has their pixels passed through a flat histogram, then a K-Means model with X clusters is created seeking for their centroids.
After that, it writes those images from each cluster in their respective cluster directories. Resulting in images separated by their simmilarities. 


it's not 100% perfect, but it helps a lot in separating contents from a given directory. 



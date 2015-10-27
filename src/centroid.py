import numpy


class Centroid():
    def __init__(self):
        pass

    def getCentroid(self, X):
        dim = len(X[0])
        centr = [0] * dim
        for i in X:
            centr = [i + j for i, j in zip(centr, i)]
        return [float(i) / float(dim) for i in centr]

    def getDistancesToCentroid(self, X, centroid):
        dists = []
        centroidArray = numpy.array(centroid)
        for x in X:
            dists.append(numpy.linalg.norm(centroidArray - numpy.array(x)))

        return dists

    def getNFarthestPoints(self, X, centroid, n):
        dists = self.getDistancesToCentroid(X, centroid)
        return list(numpy.argsort(dists)[-1:-1 * n:-1])

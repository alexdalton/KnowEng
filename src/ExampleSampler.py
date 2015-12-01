from helpers import helpers
import numpy as np
from random import sample, random


class ExampleSampler:
    def __init__(self, dict_X, dict_y, logger):
        self.helpersObj = helpers(logger)
        self.logger = logger
        self.dict_X = dict_X
        self.dict_y = dict_y

    def randomUnderSample(self, keys, N):
        numRemove = int(len(keys) * N)
        randomRemoveKeys = sample(keys, numRemove)
        self.logger.log("Random Undersampling by N = {0}".format(N))
        for removeKey in randomRemoveKeys:
            del self.dict_X[removeKey]
            del self.dict_y[removeKey]
        self.logger.log("Randomly removed {0} samples".format(len(randomRemoveKeys)))

        return self.dict_X, self.dict_y, randomRemoveKeys

    def tomekUnderSample(self, minorityKeys, majorityKeys):
        self.logger.log("Undersampling majority set by removing Tomek Links")
        tomekLinkKeys = []
        for minorityKey in minorityKeys:
            minorityExample = self.dict_X[minorityKey]
            for majorityKey in majorityKeys:
                majorityExample = self.dict_X[majorityKey]
                distance = np.linalg.norm(np.array(minorityExample) - np.array(majorityExample))
                isTomek = False

                for otherKey in set(minorityKeys) | set(majorityKeys):
                    if otherKey == minorityKey or otherKey == majorityKey:
                        continue
                    otherExample = self.dict_X[otherKey]
                    otherLabel = self.dict_y[otherKey]
                    if otherLabel == 1:
                        otherDistance = np.linalg.norm(np.array(majorityExample) - np.array(otherExample))
                    else:
                        otherDistance = np.linalg.norm(np.array(minorityExample) - np.array(otherExample))
                    if otherDistance < distance:
                        isTomek = True
                        break

                if isTomek:
                    tomekLinkKeys.append(majorityKey)

        for key in tomekLinkKeys:
            del self.dict_X[key]
            del self.dict_y[key]

        self.logger.log("Removed {0} samples".format(len(tomekLinkKeys)))

        return self.dict_X, self.dict_y, tomekLinkKeys

    def smote(self, minorityKeys, N, k, label):
        T = len(minorityKeys)

        if k > T:
            self.logger.log("k greater than the size of minority set")
            return self.dict_X, self.dict_y, []

        # N < 100 so get random sample of minorityKeys to smote N percent of them
        if N < 100:
            minorityKeys = sample(minorityKeys, int(float(N) * T / 100))

        self.logger.log("SMOTE: N = {0}, k = {1}".format(N, k))

        N = int(N / 100)
        syntheticCount = 0
        syntheticKeys = []

        for i in range(0, T):
            minorityExample = self.dict_X[minorityKeys[i]]
            distances = []
            # calculate distance from minorityExample to all other examples including self
            for j in range(0, T):
                otherMinorityExample = self.dict_X[minorityKeys[j]]
                distance = np.linalg.norm(np.array(minorityExample) - np.array(otherMinorityExample))
                distances.append(distance)
            # argsort distance and get k nearest, ignore index 0 since it's the distance with itself = 0
            kNearest = list(np.argsort(np.array(distances)))[1:k+1]

            for randomK in sample(kNearest, N):
                neighborMinorityExample = self.dict_X[minorityKeys[randomK]]
                # set synthetic feature to initially be a copy of the original
                syntheticExample = list(minorityExample)
                for featureIndex in range(0, len(minorityExample)):
                    dif = minorityExample[featureIndex] != neighborMinorityExample[featureIndex]
                    # if bits differ and with probability 50% flip the bit
                    if dif and random() > 0.5:
                        syntheticExample[featureIndex] ^= 1

                syntheticKey = "synthetic_" + str(syntheticCount)
                syntheticCount += 1
                syntheticKeys.append(syntheticKey)
                self.dict_X[syntheticKey] = syntheticExample
                self.dict_y[syntheticKey] = label

        self.logger.log("SMOTEed {0} synthetic examples from the minority set".format(syntheticCount))
        return self.dict_X, self.dict_y, syntheticKeys

    def random(self):
        pass

    def oneSidedTomek(self):
        pass
package svm

import jnisvmlight._

class svm(trainData: Array[LabeledFeatureVector], sort_input_vectors: Boolean = true, verbose: Int = 1) {
  private val trainer =  new SVMLightInterface()
  private val trainingParameters = new TrainingParameters()

  SVMLightInterface.SORT_INPUT_VECTORS = sort_input_vectors
  trainingParameters.getLearningParameters.verbosity = verbose

  private val model = trainer.trainModel(trainData, trainingParameters)

  def classify(featureVectors: Array[LabeledFeatureVector]): Seq[LabeledFeatureVector] = {
    featureVectors.foreach(vector => vector.setLabel(model.classify(vector)))
    featureVectors
  }

}

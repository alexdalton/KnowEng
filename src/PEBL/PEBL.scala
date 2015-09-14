package PEBL

// -Djava.library.path=libs/
import jnisvmlight.LabeledFeatureVector
import svm.svm

object PEBL {

  def main(args: Array[String]) {
    val positives = getPositiveFeatureVectors
    val unlabeled = getUnlabeledFeatureVectors

    var negatives: Array[LabeledFeatureVector] = Array.empty
    var newNegatives = getStrongNegatives(positives, unlabeled)

    while (!newNegatives.isEmpty) {
      negatives = negatives.diff(newNegatives).union(newNegatives)
      val iterSVM = new svm(positives ++ negatives)
      val iterP = unlabeled.diff(negatives)
      newNegatives = iterSVM.classify(iterP).filter(_.getLabel < 0).toArray
    }

  }

  def getStrongNegatives(positives: Array[LabeledFeatureVector], unlabeled: Array[LabeledFeatureVector]): Array[LabeledFeatureVector] = {
    Array.empty
  }

  def getPositiveFeatureVectors: Array[LabeledFeatureVector] = {
    Array.empty
  }

  def getUnlabeledFeatureVectors: Array[LabeledFeatureVector] = {
    Array.empty
  }

//  def getFeatureVectors(label: Double, size: Int): Array[LabeledFeatureVector] = {
//    Array.fill(size) {
//      val nDims = 1 + rd.nextInt(M - 1)
//      val hashedDims = new HashSet[Int]()
//      while (hashedDims.size() < nDims) {
//        hashedDims.add(1 + rd.nextInt(M - 1))
//      }
//
//      val iterator = hashedDims.iterator()
//      val dims = Array.fill(nDims) {
//        iterator.next()
//      }
//      val values = Array.fill(nDims) {
//        rd.nextDouble()
//      }
//
//      new LabeledFeatureVector(label, dims, values)
//    }
//  }

}
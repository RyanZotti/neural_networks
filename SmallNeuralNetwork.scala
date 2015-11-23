object Run {
  def main(args: Array[String]): Unit = {
    var errors : Seq[Double] = Seq()
    val e = 2.7182818284590452353602874713527
    val observations : List[List[Int]] = List(List(0,0,1),List(0,1,1),List(1,0,1),List(1,1,1))
    val target = List(0,1,1,0)
    var weightsHiddenNeurons = for (i <- 0 to 4) yield Seq.fill(3)(scala.util.Random.nextDouble() * 2 -1)
    var weightsOuterLayerNeuron = Seq.fill(5)(scala.util.Random.nextDouble() * 2 -1)
    for(epoch <- 0 to 60000){
      for (obsIndex <- 0 to observations.length -1){
        var hiddenLayerOutputs : List[Double] = List()
        for (neuron <- weightsHiddenNeurons){
          val dotProduct = (for (i <- 0 to observations(obsIndex).length-1) yield neuron(i) * observations(obsIndex)(i)).sum
          hiddenLayerOutputs = hiddenLayerOutputs :+ (1/(1+scala.math.pow(e,-1 * dotProduct)))
        }
        val dotProduct = (for (i <- 0 to hiddenLayerOutputs.size -1) yield weightsOuterLayerNeuron(i) * hiddenLayerOutputs(i)).sum
        val outerLayerOutput =  1/(1+scala.math.pow(e,-1 * dotProduct))
        val error = target(obsIndex) - outerLayerOutput
        errors = errors :+ Math.abs(error)
        val deltaOuterLayer = error * (outerLayerOutput * (1 - outerLayerOutput))
        for (i <- 0 to weightsOuterLayerNeuron.length -1){
          weightsOuterLayerNeuron = weightsOuterLayerNeuron.updated(i, weightsOuterLayerNeuron(i) + (deltaOuterLayer * hiddenLayerOutputs(i)))
        }
        for (neuronIndex <- 0 to weightsHiddenNeurons.length -1){
          val deltaInnerLayer = deltaOuterLayer * (hiddenLayerOutputs(neuronIndex) * (1- hiddenLayerOutputs(neuronIndex)))
          val newWeights = for (weightIndex <- 0 to weightsHiddenNeurons(neuronIndex).length -1) yield weightsHiddenNeurons(neuronIndex)(weightIndex) + deltaInnerLayer * observations(obsIndex)(weightIndex)
          weightsHiddenNeurons = (weightsHiddenNeurons.updated(neuronIndex,newWeights))
        }
      }
    }
    println(errors.sum / errors.length)
  }
}

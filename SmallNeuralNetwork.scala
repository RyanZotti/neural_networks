object Run {
  def main(args: Array[String]): Unit = {
    val e = 2.7182818284590452353602874713527
    val observations : List[List[Int]] = List(List(0,0,1),List(0,1,1),List(1,0,1),List(1,1,1))
    val target = List(0,1,1,0)
    var hiddenLayerNeurons = for (i <- 0 to 4) yield Seq.fill(3)(scala.util.Random.nextDouble())
    var outerLayerNeurons = Seq.fill(5)(scala.util.Random.nextDouble())
    for(epoch <- 0 to 6000){
      for (observation <- observations){
        var hiddenLayerOutputs : List[Double] = List()
        for (neuron <- hiddenLayerNeurons){
          var dotProduct = (for (i <- 0 to observation.length-1) yield neuron(i) * observation(i)).sum
          hiddenLayerOutputs = (hiddenLayerOutputs :+ (1/(1+scala.math.pow(e,-1 * dotProduct))))
        }
        var dotProduct = (for (i <- 0 to hiddenLayerOutputs.size -1) yield outerLayerNeurons(i) * hiddenLayerOutputs(i)).sum
        val outerLayerOutput =  (1/(1+scala.math.pow(e,-1 * dotProduct)))
        var error = target(0) - outerLayerOutput
        println(error) // See how error improves as the epochs progress
        val deltaOuterLayer = error * ((outerLayerOutput) * (1 - outerLayerOutput))
        for (i <- 0 to outerLayerNeurons.length -1){
          outerLayerNeurons = outerLayerNeurons.updated(i, outerLayerNeurons(i) + (deltaOuterLayer * hiddenLayerOutputs(i)))
        }
        for (neuronIndex <- 0 to hiddenLayerNeurons.length -1){
          val deltaInnerLayer = deltaOuterLayer * (hiddenLayerOutputs(neuronIndex) * (1- hiddenLayerOutputs(neuronIndex)))
          var newWeights : Seq[Double] = Seq()
          for (weightIndex <- 0 to hiddenLayerNeurons(neuronIndex).length -1){
            val oldVal = hiddenLayerNeurons(neuronIndex)(weightIndex)
            val newVal = oldVal + deltaInnerLayer * observation(weightIndex)
            newWeights = (newWeights :+ newVal)
          }
          hiddenLayerNeurons = (hiddenLayerNeurons.updated(neuronIndex,newWeights))
        }
      }
    }
  }
}

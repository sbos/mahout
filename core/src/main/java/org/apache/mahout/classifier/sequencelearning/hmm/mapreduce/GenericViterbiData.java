package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.io.GenericWritable;
import org.apache.hadoop.io.Writable;

/*
 */
class GenericViterbiData extends GenericWritable {
  static Class[] classes = new Class[] {
    ObservedSequenceWritable.class,
    InitialProbabilitiesWritable.class
  };

  public static GenericViterbiData fromObservedSequence(ObservedSequenceWritable sequence) {
    final GenericViterbiData data = new GenericViterbiData();
    data.set(sequence);
    return data;
  }

  public static GenericViterbiData fromInitialProbabilities(InitialProbabilitiesWritable probs) {
    final GenericViterbiData data = new GenericViterbiData();
    data.set(probs);
    return data;
  }

  @Override
  protected Class<? extends Writable>[] getTypes() {
    return classes;
  }
}

package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.io.GenericWritable;
import org.apache.hadoop.io.Writable;

/*
 */
class ForwardViterbiData extends GenericWritable {
  static Class[] classes = new Class[] {
    ObservedSequenceWritable.class,
    InitialProbabilitiesWritable.class
  };

  public ForwardViterbiData() {

  }

  public static ForwardViterbiData fromObservedSequence(ObservedSequenceWritable sequence) {
    final ForwardViterbiData data = new ForwardViterbiData();
    data.set(sequence);
    return data;
  }

  public static ForwardViterbiData fromInitialProbabilities(InitialProbabilitiesWritable probs) {
    final ForwardViterbiData data = new ForwardViterbiData();
    data.set(probs);
    return data;
  }

  @Override
  protected Class<? extends Writable>[] getTypes() {
    return classes;
  }
}

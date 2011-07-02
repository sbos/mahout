package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.io.GenericWritable;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.VarIntWritable;

/*
 */
class ViterbiDataWritable extends GenericWritable {
  static Class[] classes = new Class[] {
    ObservedSequenceWritable.class,
    HiddenStateProbabilitiesWritable.class,
    VarIntWritable.class, BackpointersWritable.class
  };

  public ViterbiDataWritable() {

  }

  public ViterbiDataWritable(int value) {
    set(new VarIntWritable(value));
  }

  public ViterbiDataWritable(BackpointersWritable backpointers) {
    set(backpointers);
  }

  public static ViterbiDataWritable fromObservedSequence(ObservedSequenceWritable sequence) {
    ViterbiDataWritable data = new ViterbiDataWritable();
    data.set(sequence);
    return data;
  }

  public static ViterbiDataWritable fromInitialProbabilities(HiddenStateProbabilitiesWritable probs) {
    ViterbiDataWritable data = new ViterbiDataWritable();
    data.set(probs);
    return data;
  }

  @Override
  protected Class<? extends Writable>[] getTypes() {
    return classes;
  }
}

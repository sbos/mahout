package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.io.GenericWritable;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.VarIntWritable;

class BackwardViterbiData extends GenericWritable {
  public BackwardViterbiData() {
  }

  public BackwardViterbiData(int state) {
    set(new VarIntWritable(state));
  }

  public BackwardViterbiData(BackpointersWritable backpointers) {
    set(backpointers);
  }

  private static final Class[] classes = new Class[] {
    VarIntWritable.class, BackpointersWritable.class
  };

  @Override
  protected Class<? extends Writable>[] getTypes() {
    return classes;
  }
}

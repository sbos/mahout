package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.mahout.math.VarIntWritable;

public class HiddenSequenceWritable extends ArrayWritable {
  public HiddenSequenceWritable() {
    super(VarIntWritable.class);
  }

  public HiddenSequenceWritable(VarIntWritable[] sequence) {
    this();
    set(sequence);
  }
}

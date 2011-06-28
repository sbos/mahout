package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Writable;

/**
 */
class InitialProbabilitiesWritable extends ArrayWritable {
  public InitialProbabilitiesWritable() {
    super(DoubleWritable.class);
  }

  public InitialProbabilitiesWritable(double[] probabilities) {
    super(DoubleWritable.class);
    final Writable[] values = new Writable[probabilities.length];
    for (int i = 0; i < probabilities.length; ++i)
      values[i] = new DoubleWritable(probabilities[i]);
    set(values);
  }

  public double[] toProbabilityArray() {
    Writable[] values = get();
    double[] probabilities = new double[values.length];

    for (int i = 0; i < values.length; ++i)
      probabilities[i] = ((DoubleWritable)values[i]).get();

    return probabilities;
  }

  public int getMostProbableState() {
    final Writable[] data = get();
    int maxState = 0;
    for (int i = 0; i < data.length; ++i) {
      if (((DoubleWritable)data[i]).get() < ((DoubleWritable)data[maxState]).get())
        maxState = i;
    }
    return maxState;
  }
}

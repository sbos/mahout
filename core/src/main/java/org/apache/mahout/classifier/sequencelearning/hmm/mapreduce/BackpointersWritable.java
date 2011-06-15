package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.commons.lang.NullArgumentException;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.VarIntWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

class BackpointersWritable implements Writable {
  int[][] backpointers;

  public BackpointersWritable() {
    backpointers = new int[1][1];
  }

  public BackpointersWritable(int nOfObservations, int nOfHiddenStates) {
    backpointers = new int[nOfObservations][nOfHiddenStates];
  }

  public BackpointersWritable(int[][] backpointers) {
    if (backpointers == null)
      throw new NullArgumentException("backpointers");
    this.backpointers = backpointers;
  }

  public int getNumberOfObservations() {
    return backpointers.length;
  }

  public int getNumberOfHiddenStates() {
    return backpointers[0].length;
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    final VarIntWritable value = new VarIntWritable(backpointers.length);
    value.write(dataOutput);
    value.set(backpointers[0].length);
    value.write(dataOutput);
    for (int i = 0; i < backpointers.length; ++i) {
      for (int j = 0; j < backpointers[i].length; ++j) {
        value.set(backpointers[i][j]);
        value.write(dataOutput);
      }
    }
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    final VarIntWritable value = new VarIntWritable();
    value.readFields(dataInput);
    final int nOfObservations = value.get();
    value.readFields(dataInput);
    final int nOfHiddenStates = value.get();
    backpointers = new int[nOfObservations][nOfHiddenStates];
    for (int i = 0; i < backpointers.length; ++i) {
      for (int j = 0; j < backpointers[i].length; ++j) {
        value.readFields(dataInput);
        backpointers[i][j] = value.get();
      }
    }
  }
}

package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.commons.lang.NullArgumentException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.VarIntWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

class BackpointersWritable implements Writable {
  int[][] backpointers;
  int chunkNumber = -1;

  public BackpointersWritable() {
    backpointers = null;
  }

  public BackpointersWritable(int[][] backpointers, int chunkNumber) {
    if (backpointers == null)
      throw new NullArgumentException("backpointers");
    this.backpointers = backpointers;
    setChunkNumber(chunkNumber);
  }

  public void setChunkNumber(int value) {
    if (value < 0)
      throw new IllegalArgumentException("value < 0");
    chunkNumber = value;
  }

  public int getChunkNumber() {
    return chunkNumber;
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    VarIntWritable value = new VarIntWritable(backpointers.length);
    value.write(dataOutput);
    value.set(backpointers[0].length);
    value.write(dataOutput);
    for (int i = 0; i < backpointers.length; ++i) {
      for (int j = 0; j < backpointers[i].length; ++j) {
        value.set(backpointers[i][j]);
        value.write(dataOutput);
      }
    }
    IntWritable chunk = new IntWritable(getChunkNumber());
    chunk.write(dataOutput);
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    VarIntWritable value = new VarIntWritable();
    value.readFields(dataInput);
    int nOfObservations = value.get();
    value.readFields(dataInput);
    int nOfHiddenStates = value.get();
    backpointers = new int[nOfObservations][nOfHiddenStates];
    for (int i = 0; i < backpointers.length; ++i) {
      for (int j = 0; j < backpointers[i].length; ++j) {
        value.readFields(dataInput);
        backpointers[i][j] = value.get();
      }
    }
    IntWritable chunk = new IntWritable();
    chunk.readFields(dataInput);
    setChunkNumber(chunk.get());
  }
}

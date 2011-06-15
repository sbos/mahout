package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.math.VarIntWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

class SequenceKey implements WritableComparable<SequenceKey> {
  private Text name;
  private VarIntWritable chunkNumber;

  public SequenceKey() {
    name = new Text();
    chunkNumber = new VarIntWritable();
  }

  public SequenceKey(String name, int chunkNumber) {
    this.name = new Text(name);
    this.chunkNumber = new VarIntWritable(chunkNumber);
  }

  public SequenceKey next() {
    final SequenceKey next = new SequenceKey(name.toString(), chunkNumber.get()+1);
    return next;
  }

  public String getSequenceName() {
    return name.toString();
  }

  public int getChunkNumber() {
    return chunkNumber.get();
  }

  @Override
  public int compareTo(SequenceKey sequenceKey) {
    return name.compareTo(sequenceKey.name);
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    name.write(dataOutput);
    chunkNumber.write(dataOutput);
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    name.readFields(dataInput);
    chunkNumber.readFields(dataInput);
  }

  @Override
  public String toString() {
    return name.toString() + "/" + chunkNumber;
  }
}

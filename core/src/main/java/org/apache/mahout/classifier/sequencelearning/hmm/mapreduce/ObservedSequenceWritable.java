package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.commons.lang.NullArgumentException;
import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.math.VarIntWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

/**
 * The class for modeling a sequence of observed variable for parallel functionality in MapReduce
 */
public class ObservedSequenceWritable implements WritableComparable<ObservedSequenceWritable>, Cloneable {
  private int[] data;
  private int length = 0;

  public ObservedSequenceWritable() {
    length = 0;
    data = null;
  }

  public ObservedSequenceWritable(int length) {
    setLength(length);
  }

  public ObservedSequenceWritable(int[] data) {
    setData(data);
  }

  public ObservedSequenceWritable(int[] data, int length) {
    this.data = data;
    this.length = length;
  }

  @Override
  public ObservedSequenceWritable clone() {
    return new ObservedSequenceWritable(data);
  }

  public void assign(int value) {
    Arrays.fill(data, value);
  }

  public int[] getData() {
    return data;
  }

  public void setData(int[] data) {
    if (data == null)
      throw new NullArgumentException("data");
    this.data = data;
    this.length = data.length;
  }

  public int getLength() {
    return length;
  }

  public void setLength(int length) {
    int[] newData = new int[length];
    this.length = length;
    if (data == null) {
      data = newData;
      return;
    }
    System.arraycopy(data, 0, newData, 0, Math.min(length, data.length));
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    VarIntWritable value = new VarIntWritable(getLength());
    value.write(dataOutput);
    for (int i = 0; i < getLength(); ++i) {
      value.set(data[i]);
      value.write(dataOutput);
    }
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    VarIntWritable value = new VarIntWritable();
    value.readFields(dataInput);
    int length = value.get();
    setLength(length);
    for (int i = 0; i < length; ++i) {
      value.readFields(dataInput);
      data[i] = value.get();
    }
  }

  @Override
  public int compareTo(ObservedSequenceWritable observedSequenceWritable) {
    int lengthDifference = getLength() - observedSequenceWritable.getLength();
    if (lengthDifference != 0)
      return lengthDifference;
    int[] otherData = observedSequenceWritable.getData();
    for (int i = 0; i < getLength(); ++i) {
      int difference = data[i] - otherData[i];
      if (difference != 0)
        return difference;
    }
    return 0;
  }

  @Override
  public boolean equals(Object other) {
    return (other instanceof ObservedSequenceWritable) ? (compareTo((ObservedSequenceWritable)other) == 0) : false;
  }

  @Override
  public int hashCode() {
    int hash = ((Integer)getLength()).hashCode();
    for (int i = 0; i < data.length; ++i)
      hash += i * ((Integer)data[i]).hashCode();
    return hash;
  }
}

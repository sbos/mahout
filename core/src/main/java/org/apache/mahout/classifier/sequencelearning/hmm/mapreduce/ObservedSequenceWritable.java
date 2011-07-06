/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.commons.lang.NullArgumentException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.math.VarIntWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

/**
 * The class for modeling a sequence of observed variables
 */
public class ObservedSequenceWritable implements WritableComparable<ObservedSequenceWritable>, Cloneable {
  private int[] data;
  private int length = 0;
  private int chunkNumber = -1;

  public ObservedSequenceWritable() {
    length = 0;
    data = null;
  }

  public ObservedSequenceWritable(int length, int chunkNumber) {
    setLength(length);
    setChunkNumber(chunkNumber);
  }

  public ObservedSequenceWritable(int[] data, int chunkNumber) {
    setData(data);
    setChunkNumber(chunkNumber);
  }

  public ObservedSequenceWritable(int[] data, int length, int chunkNumber) {
    this.data = data;
    this.length = length;
    setChunkNumber(chunkNumber);
  }

  @Override
  public ObservedSequenceWritable clone() {
    return new ObservedSequenceWritable(data, chunkNumber);
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

  public int getChunkNumber() {
    return chunkNumber;
  }

  public void setChunkNumber(int value) {
    if (value < 0)
      throw new IllegalArgumentException("value");
    chunkNumber = value;
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    VarIntWritable value = new VarIntWritable(getLength());
    value.write(dataOutput);
    for (int i = 0; i < getLength(); ++i) {
      value.set(data[i]);
      value.write(dataOutput);
    }
    IntWritable number = new IntWritable(chunkNumber);
    number.write(dataOutput);
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
    IntWritable number = new IntWritable();
    number.readFields(dataInput);
    setChunkNumber(number.get());
  }

  @Override
  public int compareTo(ObservedSequenceWritable observedSequenceWritable) {
    int lengthDifference = getLength() - observedSequenceWritable.getLength();
    if (lengthDifference != 0)
      return lengthDifference;
    int chunkDifference = chunkNumber - observedSequenceWritable.chunkNumber;
    if (chunkDifference != 0)
      return chunkDifference;
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
      hash += (i+1) * ((Integer)data[i]).hashCode();
    return hash + Integer.valueOf(chunkNumber).hashCode();
  }
}

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
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.VarIntWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Class for modeling temporary back-pointers array for parallel Viterbi functionality
 */
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

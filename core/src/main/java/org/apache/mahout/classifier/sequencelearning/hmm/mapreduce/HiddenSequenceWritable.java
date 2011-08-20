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

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.list.AbstractIntList;
import org.apache.mahout.math.list.IntArrayList;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Class for modeling sequence of decoded hidden variables
 * It's used to write the output of {@link ParallelViterbiDriver} tasks
 * Actually {@link BackwardViterbiReducer} writes them as the side effect
 */
public class HiddenSequenceWritable implements Writable {
  private AbstractIntList sequence;

  public HiddenSequenceWritable() {
    sequence = new IntArrayList();
  }

  public HiddenSequenceWritable(int[] sequence) {
    this.sequence = new IntArrayList(sequence);
  }

  public HiddenSequenceWritable(AbstractIntList list) {
    sequence = list;
  }

  @Override
  public void write(DataOutput output) throws IOException {
    if (sequence == null)
      throw new IllegalStateException("Sequence was not initialized");
    output.writeInt(sequence.size());
    VarIntWritable n = new VarIntWritable();
    for (int state: sequence.elements()) {
      n.set(state);
      n.write(output);
    }
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    int[] sequence = new int[dataInput.readInt()];
    VarIntWritable n = new VarIntWritable();
    for (int i = 0; i < sequence.length; ++i) {
      n.readFields(dataInput);
      sequence[i] = n.get();
    }
    this.sequence = new IntArrayList(sequence);
  }

  public int[] get() {
    return sequence.elements();
  }
}

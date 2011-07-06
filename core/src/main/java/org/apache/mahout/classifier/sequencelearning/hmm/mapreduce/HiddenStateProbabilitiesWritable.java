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

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Writable;

/**
 * Probabilities of the assigning last observed state to corresponding hidden states.
 * {@link ArrayWritable} of {@link DoubleWritable}
 */
class HiddenStateProbabilitiesWritable extends ArrayWritable {
  public HiddenStateProbabilitiesWritable() {
    super(DoubleWritable.class);
  }

  public HiddenStateProbabilitiesWritable(double[] probabilities) {
    super(DoubleWritable.class);
    Writable[] values = new Writable[probabilities.length];
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
    Writable[] data = get();
    int maxState = 0;
    for (int i = 1; i < data.length; ++i) {
      if (((DoubleWritable)data[i]).get() > ((DoubleWritable)data[maxState]).get())
        maxState = i;
    }
    return maxState;
  }
}

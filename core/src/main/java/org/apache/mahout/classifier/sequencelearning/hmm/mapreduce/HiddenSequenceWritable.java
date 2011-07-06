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
import org.apache.mahout.math.VarIntWritable;

/**
 * Class for modeling sequence of decoded hidden variables
 * It's used to write the output of {@link ParallelViterbiDriver} tasks
 * Actually {@link BackwardViterbiReducer} writes them as the side effect
 */
public class HiddenSequenceWritable extends ArrayWritable {
  public HiddenSequenceWritable() {
    super(VarIntWritable.class);
  }

  public HiddenSequenceWritable(VarIntWritable[] sequence) {
    this();
    set(sequence);
  }
}

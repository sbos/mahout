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

import org.apache.hadoop.io.GenericWritable;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmOnlineViterbi;
import org.apache.mahout.math.VarIntWritable;

/**
 * Generic Writable wrapper for multiple inputs that could be used by
 * @see org.apache.mahout.classifier.sequencelearning.hmm.mapreduce.BackwardViterbiReducer and
 * @see org.apache.mahout.classifier.sequencelearning.hmm.mapreduce.ForwardViterbiReducer
 */
class ViterbiDataWritable extends GenericWritable {
  static Class[] classes = new Class[] {
    ObservedSequenceWritable.class,
    HiddenStateProbabilitiesWritable.class,
    VarIntWritable.class, BackpointersWritable.class,
    HmmOnlineViterbi.class
  };

  public ViterbiDataWritable() {

  }

  public ViterbiDataWritable(int value) {
    set(new VarIntWritable(value));
  }

  public ViterbiDataWritable(HmmOnlineViterbi onlineViterbi) {
    set(onlineViterbi);
  }

  public ViterbiDataWritable(BackpointersWritable backpointers) {
    set(backpointers);
  }

  public static ViterbiDataWritable fromObservedSequence(ObservedSequenceWritable sequence) {
    ViterbiDataWritable data = new ViterbiDataWritable();
    data.set(sequence);
    return data;
  }

  public static ViterbiDataWritable fromInitialProbabilities(HiddenStateProbabilitiesWritable probs) {
    ViterbiDataWritable data = new ViterbiDataWritable();
    data.set(probs);
    return data;
  }

  @Override
  protected Class<? extends Writable>[] getTypes() {
    return classes;
  }
}

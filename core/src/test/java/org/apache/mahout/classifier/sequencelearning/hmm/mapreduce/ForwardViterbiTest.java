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

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmAlgorithms;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmEvaluator;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmModel;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.easymock.EasyMock;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ForwardViterbiTest extends MahoutTestCase {
  private HmmModel model;
  private int[] observations;
  private int sequenceLength = 33;
  private double epsilon = 0.001;
  private int[][] phi;
  private double[][] delta;

  @Override
  public void setUp() {
    model = new HmmModel(2, 2);
    double e = 0.1;
    model.setInitialProbabilities(new DenseVector(new double[] {e, 1.0-e}));

    model.getEmissionMatrix().set(0, 0, e);
    model.getEmissionMatrix().set(0, 1, 1.0-e);
    model.getEmissionMatrix().set(1, 0, 1.0-e);
    model.getEmissionMatrix().set(1, 1, e);
    model.getTransitionMatrix().set(0, 0, 0.5);
    model.getTransitionMatrix().set(0, 1, 0.5);
    model.getTransitionMatrix().set(1, 0, 0.5);
    model.getTransitionMatrix().set(1, 1, 0.5);

    observations = HmmEvaluator.predict(model, sequenceLength);

    phi = new int[observations.length - 1][model.getNrOfHiddenStates()];
    delta = new double[observations.length][model.getNrOfHiddenStates()];
  }

  @Test
  public void test() throws IOException, InterruptedException {
    Reducer.Context ctx = EasyMock.createMock(Reducer.Context.class);
    ForwardViterbiReducer reducer = new ForwardViterbiReducer();
    reducer.setModel(model);

    List<ViterbiDataWritable> forwardInput = new ArrayList<ViterbiDataWritable>();
    forwardInput.add(ViterbiDataWritable.fromObservedSequence(new ObservedSequenceWritable(observations, 0)));

    reducer.setResultHandler(new ForwardViterbiReducer.ResultHandler() {
      @Override
      public void handle(String sequenceName, int[][] backpointers, int chunkNumber, double[] hiddenStateProbabilities) throws IOException, InterruptedException {
        int[] decoded = new int[sequenceLength];
        HmmAlgorithms.viterbiAlgorithm(decoded, delta, phi, model, observations, true);

        //testing last hidden state probabilities
        assertArrayEquals(delta[sequenceLength - 1], hiddenStateProbabilities, epsilon);

        //testing backpointers arrays
        for (int i = 0; i < sequenceLength - 1; ++i)
          assertArrayEquals(phi[i], backpointers[i]);
      }
    });
    reducer.reduce(new Text("test"), forwardInput, ctx);
  }
}

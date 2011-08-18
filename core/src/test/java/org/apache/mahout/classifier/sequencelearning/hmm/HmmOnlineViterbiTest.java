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

package org.apache.mahout.classifier.sequencelearning.hmm;

import com.google.common.base.Function;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class HmmOnlineViterbiTest extends MahoutTestCase {
  HmmModel model, badModel;
  int[] observations, badObservations;
  double likelihoodEpsilon = 0.0001;

  @Override
  public void setUp() throws Exception {
    super.setUp();

    model = new HmmModel(2, 2);
    double e = 0.01;
    model.setInitialProbabilities(new DenseVector(new double[] {e, 1.0-e}));

    model.getEmissionMatrix().set(0, 0, e);
    model.getEmissionMatrix().set(0, 1, 1.0-e);
    model.getEmissionMatrix().set(1, 0, 1.0-e);
    model.getEmissionMatrix().set(1, 1, e);
    model.getTransitionMatrix().set(0, 0, 0.5);
    model.getTransitionMatrix().set(0, 1, 0.5);
    model.getTransitionMatrix().set(1, 0, 0.5);
    model.getTransitionMatrix().set(1, 1, 0.5);

    badModel = new HmmModel(4, 3);
    e = 0.3;
    double f = 0.2;

    badModel.setInitialProbabilities(new DenseVector(new double[] {0.25, 0.25, 0.25, 0.25}));

    badModel.getEmissionMatrix().set(0, 0, e);
    badModel.getEmissionMatrix().set(0, 1, 1.0-e);
    badModel.getEmissionMatrix().set(0, 2, 0);

    badModel.getEmissionMatrix().set(1, 0, 1.0-e);
    badModel.getEmissionMatrix().set(1, 1, e);
    badModel.getEmissionMatrix().set(1, 2, 0);

    badModel.getEmissionMatrix().set(2, 0, e-f);
    badModel.getEmissionMatrix().set(2, 1, 1.0-e-f);
    badModel.getEmissionMatrix().set(2, 2, 2.0*f);

    badModel.getEmissionMatrix().set(3, 0, 1.0-e-f);
    badModel.getEmissionMatrix().set(3, 1, e-f);
    badModel.getEmissionMatrix().set(3, 2, 2.0*f);

    badModel.getTransitionMatrix().set(0, 0, 0.5);
    badModel.getTransitionMatrix().set(0, 1, 0.5);
    badModel.getTransitionMatrix().set(1, 0, 0.5);
    badModel.getTransitionMatrix().set(1, 1, 0.5);

    badModel.getTransitionMatrix().set(2, 3, 0.5);
    badModel.getTransitionMatrix().set(3, 2, 0.5);
    badModel.getTransitionMatrix().set(3, 3, 0.5);
    badModel.getTransitionMatrix().set(2, 2, 0.5);

    observations = HmmEvaluator.predict(model, 27);
    badObservations = HmmEvaluator.predict(badModel, 27);
  }

  @Test
  public void testOnline() {
    HmmOnlineViterbi onlineViterbi = new HmmOnlineViterbi(model, new Function<int[], Void>() {
      @Override
      public Void apply(int[] input) {
        //This tests "onlineness" of the algorithm. On the given model it should output decoded hidden variable
        //by each input observation
        assertEquals(1, input.length);
        return null;
      }
    });

    List<Integer> input = new ArrayList<Integer>();
    for (int x: observations)
      input.add(x);

    onlineViterbi.process(input);
    //At the end we have to ensure that online viterbi gets the same likelihood as the normal one
    assertTrue(Math.abs(Math.exp(onlineViterbi.finish()) - HmmEvaluator.modelLikelihood(model, observations, true)) < likelihoodEpsilon);
  }

  @Test
  public void testCorrectness() {
    //Tests the correctess of the algorithm on the model which could not be decoded online
    List<Integer> input = new ArrayList<Integer>();
    for (int x: badObservations)
      input.add(x);

    final List<Integer> decoded = new ArrayList<Integer>();
    HmmOnlineViterbi onlineViterbi = new HmmOnlineViterbi(badModel, new Function<int[], Void>() {
      @Override
      public Void apply(int[] input) {
        for (int i = 0; i < input.length; ++i)
          decoded.add(input[i]);
        return null;
      }
    });
    onlineViterbi.process(input);
    double logLikelihood = onlineViterbi.finish();

    int[] offlineDecoded = HmmEvaluator.decode(badModel, observations, true);

    assertEquals(offlineDecoded.length, decoded.size());

    //assertArrayEquals(HmmEvaluator.decode(badModel, observations, true), decodedArray);
    assertTrue(Math.abs(Math.exp(logLikelihood) - HmmEvaluator.modelLikelihood(badModel, badObservations, true)) < likelihoodEpsilon);
  }
}

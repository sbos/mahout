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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.mahout.common.CommandLineUtil;

import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Scanner;

/**
 * A class for EM training of HMM from console
 */
public class BaumWelchTrainer {
  public static void main(String[] args) throws IOException {
    final DefaultOptionBuilder optionBuilder = new DefaultOptionBuilder();
    final ArgumentBuilder argumentBuilder = new ArgumentBuilder();

    final Option inputOption = optionBuilder.withLongName("input").
      withDescription("Text file with space-separated integers to train on").
      withShortName("i").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("path").create()).withRequired(true).create();

    final Option outputOption = optionBuilder.withLongName("output").
      withDescription("Path trained HMM model should be serialized to").
      withShortName("o").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("path").create()).withRequired(true).create();

    final Option stateNumberOption = optionBuilder.withLongName("nrOfHiddenStates").
      withDescription("Number of hidden states").
      withShortName("nh").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("number").create()).withRequired(true).create();

    final Option observedStateNumberOption = optionBuilder.withLongName("nrOfObservedStates").
      withDescription("Number of observed states").
      withShortName("no").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("number").create()).withRequired(true).create();

    final Option epsilonOption = optionBuilder.withLongName("epsilon").
      withDescription("Convergence threshold").
      withShortName("e").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("number").create()).withRequired(true).create();

    final Option iterationsOption = optionBuilder.withLongName("max-iterations").
      withDescription("Maximum iterations number").
      withShortName("m").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("number").create()).withRequired(true).create();

    final Group optionGroup = new GroupBuilder().withOption(inputOption).
      withOption(outputOption).withOption(stateNumberOption).withOption(observedStateNumberOption).
      withOption(epsilonOption).withOption(iterationsOption).
      withName("Options").create();

    try {
      final Parser parser = new Parser();
      parser.setGroup(optionGroup);
      final CommandLine commandLine = parser.parse(args);

      final String input = (String) commandLine.getValue(inputOption);
      final String output = (String) commandLine.getValue(outputOption);

      final int nrOfHiddenStates = Integer.parseInt((String) commandLine.getValue(stateNumberOption));
      final int nrOfObservedStates = Integer.parseInt((String) commandLine.getValue(observedStateNumberOption));

      final double epsilon = Double.parseDouble((String) commandLine.getValue(epsilonOption));
      final int maxIterations = Integer.parseInt((String) commandLine.getValue(iterationsOption));

      //constructing random-generated HMM
      final HmmModel model = new HmmModel(nrOfHiddenStates, nrOfObservedStates, new Date().getTime());
      final List<Integer> observations = new ArrayList<Integer>();

      //reading observations
      final FileInputStream inputStream = new FileInputStream(input);
      final Scanner scanner = new Scanner(inputStream);

      while (scanner.hasNextInt()) {
        observations.add(scanner.nextInt());
      }

      scanner.close();
      inputStream.close();

      final int[] observationsArray = new int[observations.size()];
      for (int i = 0; i < observations.size(); ++i)
        observationsArray[i] = observations.get(i);

      //training
      final HmmModel trainedModel = HmmTrainer.trainBaumWelch(model,
        observationsArray, epsilon, maxIterations, true);

      //serializing trained model
      final DataOutputStream stream  = new DataOutputStream(new FileOutputStream(output));
      LossyHmmSerializer.serialize(trainedModel, stream);
      stream.close();

      //printing tranied model
      System.out.println("Initial probabilities: ");
      for (int i = 0; i < trainedModel.getNrOfHiddenStates(); ++i)
        System.out.print(i + " ");
      System.out.println();
      for (int i = 0; i < trainedModel.getNrOfHiddenStates(); ++i)
        System.out.print(trainedModel.getInitialProbabilities().get(i) + " ");
      System.out.println();

      System.out.println("Transition matrix:");
      System.out.print("  ");
      for (int i = 0; i < trainedModel.getNrOfHiddenStates(); ++i)
        System.out.print(i + " ");
      System.out.println();
      for (int i = 0; i < trainedModel.getNrOfHiddenStates(); ++i) {
        System.out.print(i + " ");
        for (int j = 0; j < trainedModel.getNrOfHiddenStates(); ++j) {
          System.out.print(trainedModel.getTransitionMatrix().get(i, j) + " ");
        }
        System.out.println();
      }
      System.out.println("Emission matrix: ");
      System.out.print("  ");
      for (int i = 0; i < trainedModel.getNrOfOutputStates(); ++i)
        System.out.print(i + " ");
      System.out.println();
      for (int i = 0; i < trainedModel.getNrOfHiddenStates(); ++i) {
        System.out.print(i + " ");
        for (int j = 0; j < trainedModel.getNrOfOutputStates(); ++j) {
          System.out.print(trainedModel.getEmissionMatrix().get(i, j) + " ");
        }
        System.out.println();
      }
    } catch (OptionException e) {
      CommandLineUtil.printHelp(optionGroup);
    }
  }
}

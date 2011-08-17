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
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.sequencelearning.hmm.mapreduce.HiddenStateProbabilitiesWritable;
import org.apache.mahout.math.DenseVector;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * Online Viterbi algorithm implementation which could decode hidden variable sequence from the given
 * sequence of observed variables as soon as some part of input sequence could be decoded. In some cases
 * this algorithm may perform at the constant space and asymptotically same time as the normal Viterbi
 * Based on Rastislav Sramek's master thesis
 * @see HmmEvaluator
 */
public class HmmOnlineViterbi implements Writable {
  private HmmModel model;
  private double[] probs;
  private Node[] leaves;
  private Node root;
  private Tree tree;
  private LinkedList<int[]> backpointers;
  private int i;
  private double lastLikelihood;
  private Function<int[], Void> output;

  /**
   * Initializes state of the algorithm with the given Hidden Markov Model
   * @param model
   */
  public HmmOnlineViterbi(HmmModel model) {
    this.model = model;
    clear();
  }

  /**
   * Initializes state of the algorithm with the given Hidden Markov Model and decoded sequence handler
   * @param model
   * @param output a {@link Function} which would be called each time some part of
   */
  public HmmOnlineViterbi(HmmModel model, Function<int[], Void> output) {
    this(model);
    setOutput(output);
  }

  public void setOutput(Function<int[], Void> output) {
    this.output = output;
  }

  public HmmModel getModel() {
    return model;
  }

  /**
   * @return log-likelihood at the end of last decoded hidden state
   */
  public double getLastLogLikelihood() {
    return lastLikelihood;
  }

  /**
   * Decode a new chunk of observed variables.
   * Note that for some models provided observations would be decoded instantly,
   * for others they could be decoded only at the end of input. You may force decoding by calling {@link org.apache.mahout.classifier.sequencelearning.hmm.HmmOnlineViterbi#finish()}
   * @param observations
   */
  public void process(Iterable<Integer> observations) {
    Iterator<Integer> iterator = observations.iterator();

    if (probs == null) {
      probs = getInitialProbabilities(model, iterator.next());
      i = 1;
    }

    while (iterator.hasNext()) {
      int observation = iterator.next();
      double[] newProbs = new double[model.getNrOfHiddenStates()];
      int[] optimalStates = new int[model.getNrOfHiddenStates()];
      Node[] newLeaves = new Node[model.getNrOfHiddenStates()];
      for (int k = 0; k < model.getNrOfHiddenStates(); ++k) {
        int maxState = -1;
        double maxProb = -Double.MAX_VALUE;
        for (int t = 0; t < model.getNrOfHiddenStates(); ++t) {
          double currentProb = getTransitionProbability(model, k, t) + probs[t];
          if (maxProb < currentProb) {
            maxProb = currentProb;
            maxState = t;
          }
        }
        optimalStates[k] = maxState;
        newProbs[k] = maxProb + getEmissionProbability(model, observation, k);

        Node node = new Node();
        node.position = i;
        node.state = k;
        node.setParent(leaves[maxState]);
        newLeaves[k] = node;
      }
      backpointers.add(optimalStates);

      tree.compress();
      Node newRoot = tree.getRoot();
      if (root != newRoot && newRoot != null) {
        lastLikelihood = newProbs[newRoot.state];
        traceback(i - newRoot.position - 1, newRoot.state, false);
        leaves = newLeaves;
        root = newRoot;
      }

      for (Node leave: newLeaves)
        tree.addLast(leave);
      probs = newProbs;
      ++i;
    }
  }

  private void clear() {
    probs = null;
    tree = new Tree();
    leaves = new Node[model.getNrOfHiddenStates()];
    root = null;
    for (int i = 0; i < model.getNrOfHiddenStates(); ++i) {
      Node node = new Node();
      node.position = 0;
      node.state = i;
      tree.addLast(node);
      leaves[i] = node;
    }

    backpointers = new LinkedList<int[]>();
    i = 0;

    lastLikelihood = -Double.MAX_VALUE;
  }

  /**
   * Forces the decoding of hidden variables using the last observed variable as the end of sequence.
   * Also resets the algorithm state which could be used for decoding again after calling this method
   * @return log-likelihood of last decoded hidden variable
   */
  public double finish() {
    int maxState = 0;
    for (int k = 1; k < model.getNrOfHiddenStates(); ++k) {
      if (probs[k] > probs[maxState])
        maxState = k;
    }

    if (backpointers.size() > 0) {
      traceback(backpointers.size(), maxState, true);
    }

    double result = probs[maxState];
    clear();
    return result;
  }

  private void traceback(int i, int state, boolean last) {
    int[] result = new int[i+1];
    result[i] = state;
    if (!last) backpointers.remove(i);
    --i;
    while (i >= 0) {
      int[] optimalStates = backpointers.get(i);
      backpointers.remove(i);
      result[i] = optimalStates[result[i+1]];
      --i;
    }

    if (output != null) output.apply(result);
  }

  @Override
  public void write(DataOutput output) throws IOException {
    HiddenStateProbabilitiesWritable probs = new HiddenStateProbabilitiesWritable(this.probs);
    probs.write(output);

    output.write(backpointers.size());
    for (int[] optimalStates: backpointers) {
      for (int state: optimalStates) {
        output.write(state);
      }
    }

    output.write(i);
    DoubleWritable doubleWritable = new DoubleWritable(lastLikelihood);
    doubleWritable.write(output);

    //serializing compressed backpointers tree
    //3 * |H| - 2 is the upper bound of node number
    HashBiMap<Node, Integer> nodeMap = HashBiMap.create(3 * model.getNrOfHiddenStates() - 2);
    root.write(nodeMap, output);
    for (Node leave: leaves)
      leave.write(nodeMap, output);

    output.write(tree.size);
    Node node = tree.first;
    while (node != null) {
      node.write(nodeMap, output);
      node = node.next;
    }
  }

  @Override
  public void readFields(DataInput input) throws IOException {
    clear();

    HiddenStateProbabilitiesWritable probs = new HiddenStateProbabilitiesWritable();
    probs.readFields(input);
    this.probs = probs.toProbabilityArray();

    int backpointersSize = input.readInt();
    for (int i = 0; i < backpointersSize; ++i) {
      int[] optimalStates = new int[model.getNrOfHiddenStates()];
      for (int j = 0; j < optimalStates.length; ++j)
        optimalStates[j] = input.readInt();
      backpointers.addLast(optimalStates);
    }

    i = input.readInt();
    DoubleWritable doubleWritable = new DoubleWritable();
    doubleWritable.readFields(input);
    lastLikelihood = doubleWritable.get();

    BiMap<Node, Integer> nodeMap = HashBiMap.create(3 * model.getNrOfHiddenStates() - 2);
    root = Node.read(nodeMap, input);

    for (int i = 0; i < leaves.length; ++i)
      leaves[i] = Node.read(nodeMap, input);

    tree.size = input.readInt();
    for (int i = 0; i < tree.size; ++i)
      tree.addLast(Node.read(nodeMap, input));
  }

  static class Node {
    public int position, state;
    public Node parent;
    public Node next;
    public Node previous;
    public int childNumber;

    public Node() {
      position = state = -1;
      parent = null;
      next = null;
      previous = null;
      childNumber = 0;
    }

    public int write(BiMap<Node, Integer> map, DataOutput output) throws IOException {
      Integer index = map.get(this);
      if (index == null) {
        output.writeBoolean(true);
        index = map.put(this, map.size());
        output.write(index);
        output.write(position);
        output.write(state);
        output.write(childNumber);
        parent.write(map, output);
      } else {
        output.writeBoolean(false);
        output.write(index);
      }

      return index;
    }

    public static Node read(BiMap<Node, Integer> map, DataInput input) throws IOException {
      boolean first = input.readBoolean();
      int index = input.readInt();
      if (first) {
        Node node = new Node();
        node.position = input.readInt();
        node.state = input.readInt();
        node.childNumber = input.readInt();
        node.parent = map.inverse().get(input.readInt());
        return node;
      }
      return map.inverse().get(index);
    }

    public void setParent(Node parent) {
      if (this.parent != null)
        --this.parent.childNumber;
      this.parent = parent;
      if (this.parent != null)
        ++this.parent.childNumber;
    }
  }

  static class Tree {
    Node first, last;
    int size;

    public Tree() {
      first = null;
      size = 0;
    }

    public void addLast(Node node) {
      if (first == null) first = node;
      if (last != null) last.next = node;
      node.next = null;
      node.previous = last;
      last = node;
      ++size;
    }

    public void remove(Node node) {
      if (node.previous != null)
        node.previous.next = node.next;
      if (node.next != null)
        node.next.previous = node.previous;

      if (first == node)
        first = first.next;
      if (last == node)
        last = last.previous;

      --size;
    }

    public Node getRoot() {
      Node node = first;
      while (node != null && node.childNumber < 2) {
        node = node.next;
      }
      return node;
    }

    public void compress() {
      Node node = first;
      while (node != null) {
        if (node.childNumber < 1) {
          if (node.parent != null) {
            --node.parent.childNumber;
          }
          remove(node);
        }
        else {
          while (node.parent != null && node.parent.childNumber == 1) {
            remove(node.parent);
            node.setParent(node.parent.parent);
          }
        }
        node = node.next;
      }
    }
  }

  private static double getTransitionProbability(HmmModel model, int i, int j) {
    return Math.log(model.getTransitionMatrix().getQuick(j, i) + Double.MIN_VALUE);
  }

  private static double getEmissionProbability(HmmModel model, int o, int h) {
    return Math.log(model.getEmissionMatrix().get(h, o) + Double.MIN_VALUE);
  }

  private static double[] getInitialProbabilities(HmmModel model, int startObservation) {
    double[] probs = new double[model.getNrOfHiddenStates()];
    for (int h = 0; h < probs.length; ++h)
      probs[h] = Math.log(model.getInitialProbabilities().getQuick(h) + Double.MIN_VALUE) +
        Math.log(model.getEmissionMatrix().getQuick(h, startObservation));
    return probs;
  }
}

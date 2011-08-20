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
import org.apache.commons.lang.NullArgumentException;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.sequencelearning.hmm.mapreduce.HiddenStateProbabilitiesWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.*;

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

  public HmmOnlineViterbi() {
    backpointers = new LinkedList<int[]>();
  }

  /**
   * Initializes state of the algorithm with the given Hidden Markov Model
   * @param model
   */
  public HmmOnlineViterbi(HmmModel model) {
    this();
    this.model = model;
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

  public int getPosition() {
    return i;
  }

  public Node getRoot() {
    return root;
  }

  public Iterable<Node> getLeaves() {
    return Arrays.asList(leaves);
  }

  public Tree getTree() {
    return tree;
  }

  public void setOutput(Function<int[], Void> output) {
    this.output = output;
  }

  public HmmModel getModel() {
    return model;
  }

  public void setModel(HmmModel model) {
    if (model == null)
      throw new NullArgumentException("model");
    this.model = model;
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
    if (model == null)
      throw new IllegalStateException("HmmModel was not initialized before decoding");

    Iterator<Integer> iterator = observations.iterator();

    if (i == 0) {
      clear();
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
    for (int k = 1; k < probs.length; ++k) {
      if (probs[k] > probs[maxState])
        maxState = k;
    }

    traceback(backpointers.size(), maxState, true);

    double result = probs[maxState];
    i = 0;
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

  static int mapNode(BiMap<Node, Integer> map, Node node) {
    if (node == null)
      return -1;
    Integer index = map.get(node);
    if (index == null) {
      index = map.size();
      map.put(node, index);
      return index;
    }
    return index;
  }

  @Override
  public void write(DataOutput output) throws IOException {
    HiddenStateProbabilitiesWritable probs = new HiddenStateProbabilitiesWritable(this.probs);
    probs.write(output);

    output.writeInt(backpointers.size());
    output.writeInt(leaves.length);
    for (int[] optimalStates: backpointers) {
      for (int state: optimalStates) {
        output.writeInt(state);
      }
    }

    output.writeInt(i);
    DoubleWritable doubleWritable = new DoubleWritable(lastLikelihood);
    doubleWritable.write(output);

    //serializing compressed backpointers tree
    //3 * |H| - 2 is the upper bound of node number
    HashBiMap<Node, Integer> nodeMap = HashBiMap.create(3 * leaves.length - 2);

    Node node = tree.first;
    while (node != null) {
      mapNode(nodeMap, node);
      node = node.next;
    }

    mapNode(nodeMap, root);
    for (Node leave: leaves)
      mapNode(nodeMap, leave);

    HashSet<Node> alreadyWritten = new HashSet<Node>();

    output.writeInt(nodeMap.size());
    for (Node entry: nodeMap.keySet())
      Node.write(nodeMap, alreadyWritten, entry, output);

    output.writeInt(tree.size);
    node = tree.first;
    while (node != null) {
      Node.write(nodeMap, alreadyWritten, node, output);
      node = node.next;
    }

    Node.write(nodeMap, alreadyWritten, root, output);

    for (Node leave: leaves)
      Node.write(nodeMap, alreadyWritten, leave, output);
  }

  @Override
  public void readFields(DataInput input) throws IOException {
    HiddenStateProbabilitiesWritable probs = new HiddenStateProbabilitiesWritable();
    probs.readFields(input);
    this.probs = probs.toProbabilityArray();

    backpointers = new LinkedList<int[]>();
    int backpointersSize = input.readInt();
    int stateNumber = input.readInt();
    for (int i = 0; i < backpointersSize; ++i) {
      int[] optimalStates = new int[stateNumber];
      for (int j = 0; j < optimalStates.length; ++j)
        optimalStates[j] = input.readInt();
      backpointers.addLast(optimalStates);
    }

    i = input.readInt();
    DoubleWritable doubleWritable = new DoubleWritable();
    doubleWritable.readFields(input);
    lastLikelihood = doubleWritable.get();

    int mapSize = input.readInt();
    BiMap<Node, Integer> nodeMap = HashBiMap.create(mapSize);
    for (int i = 0; i < mapSize; ++i)
      Node.read(nodeMap, input);

    tree = new Tree();
    int treeSize = input.readInt();
    for (int i = 0; i < treeSize; ++i)
      tree.addLast(Node.read(nodeMap, input));

    root = Node.read(nodeMap, input);

    leaves = new Node[stateNumber];
    for (int i = 0; i < leaves.length; ++i)
      leaves[i] = Node.read(nodeMap, input);
  }

  static class Node {
    int position, state;
    Node parent;
    Node next;
    Node previous;
    int childNumber;

    public Node() {
      position = state = -1;
      parent = null;
      next = null;
      previous = null;
      childNumber = 0;
    }

    public Node getNext() {
      return next;
    }

    public Node getPrevious() {
      return previous;
    }

    public Node getParent() {
      return parent;
    }

    public static int write(BiMap<Node, Integer> map, Set<Node> written, Node node, DataOutput output) throws IOException {
      if (node == null) {
        output.writeBoolean(false);
        output.writeInt(-1);
        return -1;
      }
      int index = map.get(node);
      if (!written.contains(node)) {
        output.writeBoolean(true);
        output.writeInt(index);
        output.writeInt(node.position);
        output.writeInt(node.state);
        output.writeInt(node.childNumber);
        written.add(node);
        write(map, written, node.parent, output);
      }
      else {
        output.writeBoolean(false);
        output.writeInt(index);
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
        map.put(node, index);
        node.parent = read(map, input);
        return node;
      }
      if (index == -1)
        return null;

      return map.inverse().get(index);
    }

    public void setParent(Node parent) {
      if (this.parent != null)
        --this.parent.childNumber;
      this.parent = parent;
      if (this.parent != null)
        ++this.parent.childNumber;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj instanceof Node) {
        Node node = (Node)obj;
        return (position == node.position) && (state == node.state) && (childNumber == node.childNumber)
          && (parent != null ? parent.equals(parent) : node.parent == null);
      }
      return false;
    }
  }

  public static class Tree {
    Node first, last;
    int size;

    public Tree() {
      first = null;
      size = 0;
    }

    public int getSize() {
      return size;
    }

    public Node getFirst() {
      return first;
    }

    public Node getLast() {
      return last;
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

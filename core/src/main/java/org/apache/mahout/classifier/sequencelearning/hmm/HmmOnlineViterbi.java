package org.apache.mahout.classifier.sequencelearning.hmm;

import com.google.common.base.Function;
import org.apache.mahout.math.DenseVector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class HmmOnlineViterbi {
  private HmmModel model;
  private double[] probs;
  private Node[] leaves;
  private Node root;
  private Tree tree;
  private LinkedList<int[]> backpointers;
  private int i;
  private Function<int[], Void> output;

  public HmmOnlineViterbi(HmmModel model) {
    this.model = model;
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
  }

  public HmmOnlineViterbi(HmmModel model, Function<int[], Void> output) {
    this(model);
    setOutput(output);
  }

  public void setOutput(Function<int[], Void> output) {
    this.output = output;
  }

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
        //tree.addLast(node);
      }
      backpointers.add(optimalStates);

      tree.compress();
      Node newRoot = tree.getRoot();
      if (root != newRoot && newRoot != null) {
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

  public void finish() {
    int maxState = 0;
    for (int k = 1; k < model.getNrOfHiddenStates(); ++k) {
      if (probs[k] > probs[maxState])
        maxState = k;
    }

    if (backpointers.size() > 0) {
      traceback(backpointers.size(), maxState, true);
    }
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

    output.apply(result);
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

    public Tree() {
      first = last = null;
    }

    public void addLast(Node node) {
      if (last != null) last.next = node;
      if (first == null) first = node;
      node.previous = last;
      last = node;
    }

    public void remove(Node node) {
      if (node.previous != null)
        node.previous.next = node.next;
      if (node.next != null)
        node.next.previous = node.previous;

      if (first == node)
        first = node.next;
      if (last == node)
        last = last.previous;
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

  private static HmmModel createBad() {
    HmmModel model = new HmmModel(4, 3);
    double e = 0.3, f = 0.2;

    model.setInitialProbabilities(new DenseVector(new double[] {0.25, 0.25, 0.25, 0.25}));

    model.getEmissionMatrix().set(0, 0, e);
    model.getEmissionMatrix().set(0, 1, 1.0-e);
    model.getEmissionMatrix().set(0, 2, 0);

    model.getEmissionMatrix().set(1, 0, 1.0-e);
    model.getEmissionMatrix().set(1, 1, e);
    model.getEmissionMatrix().set(1, 2, 0);

    model.getEmissionMatrix().set(2, 0, e-f);
    model.getEmissionMatrix().set(2, 1, 1.0-e-f);
    model.getEmissionMatrix().set(2, 2, 2.0*f);

    model.getEmissionMatrix().set(3, 0, 1.0-e-f);
    model.getEmissionMatrix().set(3, 1, e-f);
    model.getEmissionMatrix().set(3, 2, 2.0*f);

    model.getTransitionMatrix().set(0, 0, 0.5);
    model.getTransitionMatrix().set(0, 1, 0.5);
    model.getTransitionMatrix().set(1, 0, 0.5);
    model.getTransitionMatrix().set(1, 1, 0.5);

    model.getTransitionMatrix().set(2, 3, 0.5);
    model.getTransitionMatrix().set(3, 2, 0.5);
    model.getTransitionMatrix().set(3, 3, 0.5);
    model.getTransitionMatrix().set(2, 2, 0.5);

    return model;
  }

  public static void main(String[] args) throws IOException {
    HmmModel model = new HmmModel(2, 2);
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

    //model = LossyHmmSerializer.deserialize(new DataInputStream(new FileInputStream("../hmm.model")));
    model = createBad();

    int[] data = HmmEvaluator.predict(model, 27);
    System.out.print("Seq: ");
    for (int x: data)
      System.out.print(x + " ");
    System.out.println();
    //System.out.print("Std: ");
    int[] dec = HmmAlgorithms.viterbiAlgorithm(model, data, true);
    for (int x: dec)
      System.out.print(x + " ");
    System.out.println();
    List<Integer> shit = new ArrayList<Integer>();
    for (int x: data)
      shit.add(x);

    HmmOnlineViterbi onlineViterbi = new HmmOnlineViterbi(model, new Function<int[], Void>() {
      @Override
      public Void apply(int[] input) {
        for (int x: input) System.out.print(x + " ");
        return null;
      }
    });

    onlineViterbi.process(shit);
    onlineViterbi.finish();
  }
}

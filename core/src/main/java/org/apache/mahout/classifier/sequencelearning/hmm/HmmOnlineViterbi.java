package org.apache.mahout.classifier.sequencelearning.hmm;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.*;

public class HmmOnlineViterbi {
  public static class Node {
    public int position, state;
    public Node parentNode;
    public int childNumber;

    public Node() {
      position = state = -1;
      parentNode = null;
      childNumber = 0;
    }

    public void setParentNode(Node parent) {
      if (parentNode != parent && parentNode != null)
        --parentNode.childNumber;
      parentNode = parent;
      ++parentNode.childNumber;
    }
  }

  static void compress(LinkedList<Node> tree) {
    ListIterator<Node> iterator = tree.listIterator();
    Node node = iterator.next();
    while (iterator.hasNext()) {
      if (node.childNumber < 1) {
        if (node.parentNode != null)
          --node.parentNode.childNumber;
        iterator.remove();
      }
      else if (node.parentNode != null) {
        while (node.parentNode.childNumber == 1) {
          node.setParentNode(node.parentNode.parentNode);
        }
      }
      node = iterator.next();
    }
  }

  static int[] traceback(List<int[]> backpointers, int i, int state) {
    int[] result = new int[backpointers.size()+1];
    result[--i] = state;
    while (i > 0) {
      --i;
      int[] optimalStates = backpointers.get(i);
      backpointers.remove(i);
      result[i] = optimalStates[result[i+1]];
    }
    for (int aResult : result) System.out.print(aResult + " ");
    System.out.println();
    return result;
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
  public static Iterable<Integer> onlineViterbi(HmmModel model, Iterable<Integer> observations) {
    Iterator<Integer> iterator = observations.iterator();

    double[] probs = getInitialProbabilities(model, iterator.next());
    LinkedList<Node> tree = new LinkedList<Node>();
    Node[] leaves = new Node[model.getNrOfHiddenStates()];
    for (int i = 0; i < model.getNrOfHiddenStates(); ++i) {
      Node node = new Node();
      node.position = 0;
      node.state = i;
      tree.push(node);
      leaves[i] = node;
    }

    List<int[]> backpointers = new ArrayList<int[]>();
    int i = 1;
    int lastOutput = 0;
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
        node.setParentNode(leaves[optimalStates[k]]);
        newLeaves[k] = node;
        tree.push(node);
      }
      backpointers.add(optimalStates);

      Node oldRoot = tree.getLast();
      compress(tree);
      Node newRoot = tree.getLast();
      if (newRoot != oldRoot) {
        traceback(backpointers, newRoot.position - lastOutput, newRoot.state);
        lastOutput = i;
        leaves = newLeaves;
      }

      probs = newProbs;
      ++i;
    }

    int maxState = 0;
    for (int k = 1; k < model.getNrOfHiddenStates(); ++k) {
      if (probs[k] > probs[maxState])
        maxState = k;
    }

    if (backpointers.size() > 0)
      traceback(backpointers, i - lastOutput, maxState);
    return null;
  }

  public static void main(String[] args) throws IOException {

    HmmOnlineViterbi.onlineViterbi(LossyHmmSerializer.deserialize(new DataInputStream(new FileInputStream("../hmm.model"))),
      Arrays.asList(1,1,1,0,0,0,0,0,0,1,1,1,0,0,2,2,2,
        2, 2, 2,2, 2, 2, 3, 3, 3, 3, 3, 3 ,3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3,
        3, 3, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3));
  }
}

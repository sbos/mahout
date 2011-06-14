package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.mahout.classifier.sequencelearning.hmm.HmmModel;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

class LossyHmmModelSerializer {
  public static void serialize(HmmModel model, DataOutput output) throws IOException {
    final MatrixWritable matrix = new MatrixWritable(model.getEmissionMatrix());
    matrix.write(output);
    matrix.set(model.getTransitionMatrix());
    matrix.write(output);

    final VectorWritable vector = new VectorWritable(model.getInitialProbabilities());
    vector.write(output);
  }

  public static HmmModel deserialize(DataInput input) throws IOException {
    final MatrixWritable matrix = new MatrixWritable();
    matrix.readFields(input);
    final Matrix emissionMatrix = matrix.get();

    matrix.readFields(input);
    final Matrix transitionMatrix = matrix.get();

    final VectorWritable vector = new VectorWritable();
    vector.readFields(input);
    final Vector initialProbabilities = vector.get();

    return new HmmModel(transitionMatrix, emissionMatrix, initialProbabilities);
  }
}

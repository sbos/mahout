package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmModel;

import java.io.IOException;
import java.net.URI;
import java.util.Iterator;

class ForwardViterbiReducer extends Reducer<SequenceKey, GenericViterbiData, SequenceKey, GenericViterbiData> {
  private SequenceFile.Writer backpointersWriter;
  private HmmModel model;

  protected void setup(Reducer.Context context)
              throws IOException,
                     InterruptedException {
    final Configuration configuration = context.getConfiguration();
    final String intermediateUri = (context.getConfiguration().get("hmm.intermediate")) ;
    final int chunkNumber = configuration.getInt("hmm.chunk_number", -1);
    final String intermediate = intermediateUri + "/" + chunkNumber + "/" + context.getJobID();
    final FileSystem intermediateFileSystem = FileSystem.get(URI.create(intermediate), context.getConfiguration());
    backpointersWriter = SequenceFile.createWriter(intermediateFileSystem,
      context.getConfiguration(), new Path(intermediate, "12"), SequenceKey.class, BackpointersWritable.class);

    final String hmmModelFileName = configuration.get("hmm.model");
    for (Path cachePath: DistributedCache.getLocalCacheFiles(configuration)) {
      if (cachePath.getName().equals(hmmModelFileName))
      {
        final FileSystem cacheFileSystem = cachePath.getFileSystem(configuration);
        final FSDataInputStream modelStream = cacheFileSystem.open(cachePath);
        model = LossyHmmModelSerializer.deserialize(modelStream);
        modelStream.close();
        break;
      }
    }
  }

  protected void cleanup(Reducer.Context context)
                throws IOException,
                       InterruptedException {
    backpointersWriter.close();
  }

  public void reduce(SequenceKey key, Iterable<GenericViterbiData> values,
                     Context context) throws IOException, InterruptedException {
    final Iterator<GenericViterbiData> iterator = values.iterator();
    double[] probabilities = null;
    int[] observations = null;
    while (iterator.hasNext()) {
      final GenericViterbiData data = iterator.next();
      final Writable value = data.get();
      if (value instanceof InitialProbabilitiesWritable)
        probabilities = ((InitialProbabilitiesWritable) value).toProbabilityArray();
      else if (value instanceof ObservedSequenceWritable)
        observations = ((ObservedSequenceWritable) value).getData();
      else
        throw new IOException("Unsupported Writable provided to the reducer");
    }

    final BackpointersWritable backpointers = new BackpointersWritable(
      forward(observations, model, probabilities));
    backpointersWriter.append(key, backpointers);
    context.write(key.next(), GenericViterbiData.fromInitialProbabilities(
      new InitialProbabilitiesWritable(probabilities)));
  }

  private static int[][] forward(int[] observations, HmmModel model, double[] probs) {
    final double[] nextProbs = new double[model.getNrOfHiddenStates()];
    final int[][] backpoints = new int[observations.length][model.getNrOfHiddenStates()];

    for (int i = 0; i < observations.length; ++i) {
      for (int t = 0; t < model.getNrOfHiddenStates(); ++t) {
        int maxState = 0;
        double maxProb = 0;
        for (int h = 0; h < model.getNrOfHiddenStates(); ++h) {
          final double currentProb = Math.log(model.getTransitionMatrix().get(t, h)) + probs[h];
          if (maxProb < currentProb) {
            maxState = h;
            maxProb = currentProb;
          }
        }
        nextProbs[t] = maxProb + Math.log(model.getEmissionMatrix().get(t, observations[i]));
        backpoints[i][t] = maxState;
      }
      probs = nextProbs;
    }

    return backpoints;
  }
}

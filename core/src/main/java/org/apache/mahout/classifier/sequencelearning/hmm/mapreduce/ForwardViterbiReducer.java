package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.Iterator;

class ForwardViterbiReducer extends Reducer<SequenceKey, ForwardViterbiData, SequenceKey, ForwardViterbiData> {
  private SequenceFile.Writer backpointersWriter;
  private SequenceFile.Writer lastStatesWriter;
  private HmmModel model;

  private static Logger log = LoggerFactory.getLogger(ForwardViterbiReducer.class);

  protected void setup(Reducer.Context context)
              throws IOException,
                     InterruptedException {
    final Configuration configuration = context.getConfiguration();
    final String backpointersPath = (context.getConfiguration().get("hmm.backpointers"));
    final String lastStatesPath = (context.getConfiguration().get("hmm.laststates"));
    final FileSystem intermediateFileSystem = FileSystem.get(URI.create(backpointersPath),
      context.getConfiguration());
    backpointersWriter = SequenceFile.createWriter(intermediateFileSystem,
      context.getConfiguration(), new Path(backpointersPath, context.getJobID().toString()),
      SequenceKey.class, BackwardViterbiData.class, SequenceFile.CompressionType.RECORD);
    lastStatesWriter = SequenceFile.createWriter(intermediateFileSystem,
      context.getConfiguration(), new Path(lastStatesPath, context.getJobID().toString()),
      SequenceKey.class, BackwardViterbiData.class, SequenceFile.CompressionType.BLOCK);

    final String hmmModelFileName = configuration.get("hmm.model");
    log.info("Trying to load model with name " + hmmModelFileName);
    for (Path cachePath: DistributedCache.getLocalCacheFiles(configuration)) {
      if (cachePath.getName().endsWith(hmmModelFileName))
      {
        final DataInputStream modelStream = new DataInputStream(new FileInputStream(cachePath.toString())) ;
        model = LossyHmmModelSerializer.deserialize(modelStream);
        log.info("Model loaded");
        modelStream.close();
        break;
      }
    }

    if (model == null)
      throw new IllegalStateException("Model " + hmmModelFileName + " was not loaded");
  }

  protected void cleanup(Reducer.Context context)
                throws IOException,
                       InterruptedException {
    backpointersWriter.close();
    lastStatesWriter.close();
  }

  public void reduce(SequenceKey key, Iterable<ForwardViterbiData> values,
                     Context context) throws IOException, InterruptedException {
    log.debug("Reducing data for " + key.toString());
    final Iterator<ForwardViterbiData> iterator = values.iterator();
    double[] probabilities = null;
    int[] observations = null;
    InitialProbabilitiesWritable lastProbabilities = null;
    while (iterator.hasNext()) {
      final ForwardViterbiData data = iterator.next();
      final Writable value = data.get();
      if (value instanceof InitialProbabilitiesWritable) {
        lastProbabilities =  ((InitialProbabilitiesWritable) value);
        probabilities = lastProbabilities.toProbabilityArray();
        log.debug("Successfully read probabilities from the previous step");
      }
      else if (value instanceof ObservedSequenceWritable) {
        observations = ((ObservedSequenceWritable) value).getData();
        log.debug("Successfully read observations from the current step");
      }
      else
        throw new IOException("Unsupported Writable provided to the reducer");
    }

    if (probabilities == null) {
      log.debug("Seems like it's first chunk, so defining probabilities to null");
      probabilities = new double[model.getNrOfHiddenStates()];
    }

    if (observations == null) {
      log.debug("Seems like all observations were processed at the previous step, nothing to do");
      assert lastProbabilities != null;
      lastStatesWriter.append(key,
        new BackwardViterbiData(lastProbabilities.getMostProbableState()));
      return;
    }

    log.info("Performing forward pass on " + key.getSequenceName() + "/" + key.getChunkNumber());
    final BackpointersWritable backpointers = new BackpointersWritable(
      forward(observations, model, probabilities));
    backpointersWriter.append(key, new BackwardViterbiData(backpointers));
    context.write(key, ForwardViterbiData.fromInitialProbabilities(
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

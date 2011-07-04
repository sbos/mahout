package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmModel;
import org.apache.mahout.classifier.sequencelearning.hmm.LossyHmmSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.Iterator;

class ForwardViterbiReducer extends Reducer<Text, ViterbiDataWritable, Text, ViterbiDataWritable> {
  private SequenceFile.Writer backpointersWriter;
  private HmmModel model;

  private static Logger log = LoggerFactory.getLogger(ForwardViterbiReducer.class);

  protected void setup(Reducer.Context context)
              throws IOException,
                     InterruptedException {
    Configuration configuration = context.getConfiguration();
    String backpointersPath = (context.getConfiguration().get("hmm.backpointers"));
    FileSystem intermediateFileSystem = FileSystem.get(URI.create(backpointersPath),
      context.getConfiguration());
    backpointersWriter = SequenceFile.createWriter(intermediateFileSystem,
      context.getConfiguration(), new Path(backpointersPath, context.getJobID().toString()),
      Text.class, ViterbiDataWritable.class, SequenceFile.CompressionType.RECORD);

    String hmmModelFileName = configuration.get("hmm.model");
    log.info("Trying to load model with name " + hmmModelFileName);
    for (Path cachePath: DistributedCache.getLocalCacheFiles(configuration)) {
      if (cachePath.getName().endsWith(hmmModelFileName))
      {
        DataInputStream modelStream = new DataInputStream(new FileInputStream(cachePath.toString())) ;
        model = LossyHmmSerializer.deserialize(modelStream);
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
  }

  @Override
  public void reduce(Text key, Iterable<ViterbiDataWritable> values,
                     Context context) throws IOException, InterruptedException {
    log.debug("Reducing data for " + key.toString());
    Iterator<ViterbiDataWritable> iterator = values.iterator();
    double[] probabilities = null;
    int[] observations = null;
    int chunkNumber = -1;
    HiddenStateProbabilitiesWritable lastProbabilities;
    while (iterator.hasNext()) {
      ViterbiDataWritable data = iterator.next();
      Writable value = data.get();
      if (value instanceof HiddenStateProbabilitiesWritable) {
        lastProbabilities =  ((HiddenStateProbabilitiesWritable) value);
        probabilities = lastProbabilities.toProbabilityArray();
        log.debug("Successfully read probabilities from the previous step");
      }
      else if (value instanceof ObservedSequenceWritable) {
        observations = ((ObservedSequenceWritable) value).getData();
        chunkNumber = ((ObservedSequenceWritable) value).getChunkNumber();
        log.debug("Successfully read observations from the current step");
      }
      else
        throw new IOException("Unsupported Writable provided to the reducer");
    }

    if (observations == null) {
      log.debug("Seems like everything is processed already, skipping this sequence");
      return;
    }
    if (probabilities == null) {
      if (chunkNumber != 0)
        throw new IllegalStateException("No hidden state probabilities were provided, but chunk number is not 0");
      log.debug("Seems like it's first chunk, so defining probabilities to initial");
      probabilities = getInitialProbabilities(model, observations[0]);
    }

    if (chunkNumber < 0)
      throw new IllegalStateException("Chunk number was not initialized");

    log.info("Performing forward pass on " + key + "/" + chunkNumber);
    BackpointersWritable backpointers = new BackpointersWritable(
      forward(observations, model, probabilities), chunkNumber);
    backpointersWriter.append(key, new ViterbiDataWritable(backpointers));
    context.write(key, ViterbiDataWritable.fromInitialProbabilities(
      new HiddenStateProbabilitiesWritable(probabilities)));
  }

  private static double[] getInitialProbabilities(HmmModel model, int startObservation) {
    double[] probs = new double[model.getNrOfHiddenStates()];
    for (int h = 0; h < probs.length; ++h)
      probs[h] = Math.log(model.getInitialProbabilities().getQuick(h) + Double.MIN_VALUE) +
        Math.log(model.getEmissionMatrix().getQuick(h, startObservation));
    return probs;
  }

  private static double getTransitionProbability(HmmModel model, int i, int j) {
    return Math.log(model.getTransitionMatrix().getQuick(j, i) + Double.MIN_VALUE);
  }

  private static double getEmissionProbability(HmmModel model, int o, int h) {
    return Math.log(model.getEmissionMatrix().get(h, o) + Double.MIN_VALUE);
  }

  private static int[][] forward(int[] observations, HmmModel model, double[] probs) {
    double[] nextProbs = new double[model.getNrOfHiddenStates()];
    int[][] backpoints = new int[observations.length - 1][model.getNrOfHiddenStates()];

    for (int i = 1; i < observations.length; ++i) {
      for (int t = 0; t < model.getNrOfHiddenStates(); ++t) {
        int maxState = 0;
        double maxProb = -Double.MAX_VALUE;
        for (int h = 0; h < model.getNrOfHiddenStates(); ++h) {
          double currentProb = getTransitionProbability(model, t, h) + probs[h];
          if (maxProb < currentProb) {
            maxState = h;
            maxProb = currentProb;
          }
        }
        nextProbs[t] = maxProb + getEmissionProbability(model, observations[i], t);
        backpoints[i - 1][t] = maxState;
      }
      for (int t = 0; t < probs.length; ++t)
        probs[t] = nextProbs[t];
    }

    return backpoints;
  }

  public static void main(String[] args) throws IOException {
    String model = args[0];
    Configuration configuration = new Configuration();
    configuration.addResource(new Path("/opt/hadoop/conf", "core-site.xml"));
    FileSystem fs = FileSystem.get(URI.create(model), configuration);
    HmmModel hmmModel = LossyHmmSerializer.deserialize(fs.open(new Path(model)));
    int[] observations  = new int[] {1, 1, 1, 0,0, 0, 0, 0, 0, 1, 1,
      1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
      2, 2, 2, 2, 3, 3, 3, 3, 3,
      2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3
};
    double[] probs = getInitialProbabilities(hmmModel, observations[0]);

    int [][] backpointers = forward(observations, hmmModel, probs);
  }
}

package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.VarIntWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;

class BackwardViterbiReducer extends Reducer<Text, ViterbiDataWritable, Text, ViterbiDataWritable> {
  private String path;

  private static final Logger log = LoggerFactory.getLogger(BackwardViterbiReducer.class);

  @Override
  protected void setup(Reducer.Context context)
              throws IOException,
                     InterruptedException {
    path = context.getConfiguration().get("hmm.output");
  }

  @Override
  public void reduce(Text key, Iterable<ViterbiDataWritable> values,
                     Context context) throws IOException, InterruptedException {
    Configuration configuration = context.getConfiguration();
    FileSystem fs = FileSystem.get(URI.create(path), configuration);
    MapFile.Writer mapWriter = new MapFile.Writer(configuration, fs, path + "/" + key,
      IntWritable.class, HiddenSequenceWritable.class);

    int[][] backpointers = null;
    int lastState = -1;
    int chunkNumber = -1;

    for (ViterbiDataWritable data: values) {
      if (data.get() instanceof BackpointersWritable) {
        backpointers = ((BackpointersWritable) data.get()).backpointers;
        chunkNumber = ((BackpointersWritable) data.get()).getChunkNumber();
      }
      else if (data.get() instanceof VarIntWritable) {
        lastState = ((VarIntWritable) data.get()).get();
      }
      else if (data.get() instanceof HiddenStateProbabilitiesWritable) {
        if (lastState == -1)
          lastState = ((HiddenStateProbabilitiesWritable) data.get()).getMostProbableState();
      }
      else {
        throw new IOException("Unsupported backward data provided");
      }
    }

    log.info("Performing backward Viterbi pass on " + key + " / " + chunkNumber);

    if (backpointers == null && lastState != -1) {
      log.info("No backpointers provided, but last state was computed from probabilities");
      context.write(key, new ViterbiDataWritable(lastState));
      return;
    }
    else if (backpointers == null)
      throw new IllegalStateException("Backpointers array was not provided to the reducer");

    if (lastState < 0)
      throw new IllegalStateException("Last state was not initialized");
    if (chunkNumber < 0)
      throw new IllegalStateException("Chunk number was not initialized");

    log.info("last state: " + lastState);
    int chunkLength = backpointers.length + 1;
    VarIntWritable[] path = new VarIntWritable[chunkLength];
    path[chunkLength - 1] = new VarIntWritable(lastState);
    for (int i = chunkLength-2; i >= 0; --i) {
      path[i] = new VarIntWritable(backpointers[i][path[i+1].get()]);
    }

    mapWriter.append(new IntWritable(-chunkNumber), new HiddenSequenceWritable(path));
    mapWriter.close();

    context.write(key, new ViterbiDataWritable(path[0].get()));
    log.info("new last state: " + path[0].get());
  }
}

package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapFile;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.VarIntWritable;

import java.io.IOException;
import java.net.URI;

class BackwardViterbiReducer extends Reducer<SequenceKey, ViterbiDataWritable, SequenceKey, ViterbiDataWritable> {
  private String path;

  @Override
  protected void setup(Reducer.Context context)
              throws IOException,
                     InterruptedException {
    path = context.getConfiguration().get("hmm.output");
  }

  @Override
  public void reduce(SequenceKey key, Iterable<ViterbiDataWritable> values,
                     Context context) throws IOException, InterruptedException {
    final Configuration configuration = context.getConfiguration();
    final String sequenceName = key.getSequenceName();
    final FileSystem fs = FileSystem.get(URI.create(path), configuration);
    final MapFile.Writer sequenceWriter = new MapFile.Writer(configuration, fs, path + "/" + sequenceName,
      IntWritable.class, HiddenSequenceWritable.class);

    int[][] backpointers = null;
    int lastState = -1;

    for (ViterbiDataWritable data: values) {
      if (data.get() instanceof BackpointersWritable) {
        backpointers = ((BackpointersWritable) data.get()).backpointers;
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

    if (backpointers == null && lastState != -1) {
      context.write(key.previous(), new ViterbiDataWritable(lastState));
      return;
    }
    else if (backpointers == null)
      throw new IllegalStateException("Backpointers array was not provided to the reducer");

    if (lastState < 0)
      throw new IllegalStateException("Last state was not initialized");

    final int chunkLength = backpointers.length;
    final VarIntWritable[] path = new VarIntWritable[chunkLength];
    for (int i = 0; i < path.length; ++i)
      path[i] = new VarIntWritable();
    path[chunkLength - 1].set(lastState);
    for (int i = path.length-2; i >= 0; --i) {
      path[i].set(backpointers[i][path[i+1].get()]);
    }

    sequenceWriter.append(new IntWritable(key.getChunkNumber()), new HiddenSequenceWritable(path));
    sequenceWriter.close();

    context.write(new SequenceKey(sequenceName, key.getChunkNumber() - 1),
      new ViterbiDataWritable(path[0].get()));
  }
}

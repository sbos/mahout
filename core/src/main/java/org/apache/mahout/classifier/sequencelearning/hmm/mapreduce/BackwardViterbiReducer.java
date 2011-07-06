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

package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.VarIntWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;

/**
 * Takes as the input {@link VarIntWritable} (last decoded hidden state from the previous step) and
 * {@link BackpointersWritable}, and produces as the output {@link VarIntWritable} (last decoded hidden state
 * for the next step). Also it writes decoded {@link HiddenSequenceWritable} in the background
 */
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

    FileSystem fs = FileSystem.get(URI.create(this.path), configuration);
    FSDataOutputStream outputStream = fs.create(new Path(this.path + "/" + key, String.valueOf(chunkNumber)));

    new HiddenSequenceWritable(path).write(outputStream);
    outputStream.close();

    context.write(key, new ViterbiDataWritable(path[0].get()));
    log.info("new last state: " + path[0].get());
  }
}

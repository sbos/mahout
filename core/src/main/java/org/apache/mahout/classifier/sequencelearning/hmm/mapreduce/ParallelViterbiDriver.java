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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.CommandLineUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;

/**
 * The driver for parallel Viterbi processing
 * It takes as the input chunks divided by {@link org.apache.mahout.classifier.sequencelearning.hmm.mapreduce.PrepareChunks}
 * and writes decoded chunks that could me merged again with
 * {@link org.apache.mahout.classifier.sequencelearning.hmm.mapreduce.PrepareChunks}
 *
 * @see org.apache.mahout.classifier.sequencelearning.hmm.mapreduce.HiddenSequenceWritable
 */
public class ParallelViterbiDriver extends Configured implements Tool {
  private String inputs;
  private String output, intermediate, model;

  private static Logger logger = LoggerFactory.getLogger(ParallelViterbiDriver.class);

  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    DefaultOptionBuilder optionBuilder = new DefaultOptionBuilder();
    ArgumentBuilder argumentBuilder = new ArgumentBuilder();

    Option inputOption = optionBuilder.withLongName("input").
      withShortName("i").withRequired(true).
      withArgument(argumentBuilder.withMinimum(1).withMaximum(1).withName("path").create()).create();

    Option outputOption = optionBuilder.withLongName("output").
      withShortName("o").withRequired(true).
      withArgument(argumentBuilder.withMinimum(1).withMaximum(1).
        withName("path").create()).create();

    Option modelOption = optionBuilder.withLongName("model").
      withShortName("m").withRequired(true).
      withArgument(argumentBuilder.withMinimum(1).withMaximum(1).
        withName("path").create()).withDescription("Serialized HMM model").create();

    Option intermediateOption = optionBuilder.withLongName("intermediate").
      withShortName("im").withRequired(true).
      withArgument(argumentBuilder.withMinimum(1).withMaximum(1).
        withName("path").withDefault("/tmp").create()).create();

    Group options = new GroupBuilder().withOption(inputOption).
      withOption(outputOption).withOption(intermediateOption).
      withOption(modelOption).withName("Options").create();

    try {
      Parser parser = new Parser();
      parser.setGroup(options);
      CommandLine commandLine = parser.parse(args);

      inputs = (String) commandLine.getValue(inputOption);
      output = (String) commandLine.getValue(outputOption);
      intermediate = (String) commandLine.getValue(intermediateOption);
      model = (String) commandLine.getValue(modelOption);

      runForward();
      runBackward();
    } catch (OptionException e) {
      CommandLineUtil.printHelp(options);
      return 1;
    }
    return 0;
  }

  private int getChunkCount() throws IOException {
    FileSystem fs = FileSystem.get(URI.create(inputs), getConf());
    Path path = new Path(inputs);
    return fs.listStatus(path).length;
  }

  private void runBackward() throws IOException, ClassNotFoundException, InterruptedException {
    logger.info("Running backward Viterbi pass");

    int chunkCount = getChunkCount();
    for (int i = chunkCount-1; i >= 0; --i) {
      logger.info("Processing chunk " + (i+1) + "/" + chunkCount);
      Job job = new Job(getConf(), "viterbi-backward-" + i);
      job.setMapperClass(Mapper.class);
      job.setReducerClass(BackwardViterbiReducer.class);
      job.setInputFormatClass(SequenceFileInputFormat.class);
      job.setOutputKeyClass(Text.class);
      job.setOutputValueClass(ViterbiDataWritable.class);
      job.setOutputFormatClass(SequenceFileOutputFormat.class);
      job.setMapOutputKeyClass(Text.class);
      job.setMapOutputValueClass(ViterbiDataWritable.class);

      SequenceFileOutputFormat.setCompressOutput(job, true);
      SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK);

      job.setJarByClass(BackwardViterbiReducer.class);

      FileOutputFormat.setOutputPath(job, getLastStatePath(i));
      FileInputFormat.addInputPath(job, getBackpointersPath(i));
      if (i < chunkCount-1)
        FileInputFormat.addInputPath(job, getLastStatePath(i+1));
      FileInputFormat.addInputPath(job, getProbabilitiesPath(i));
      Configuration jobConfiguration = job.getConfiguration();
      jobConfiguration.set("hmm.output", output);

      job.submit();
      job.waitForCompletion(true);
    }
  }

  private Path getBackpointersPath(int chunkNumber) {
    return new Path(intermediate, "backpointers/" + chunkNumber);
  }

  private Path getLastStatePath(int chunkNumber) {
    return new Path(intermediate, "laststates/" + chunkNumber);
  }

  private Path getProbabilitiesPath(int chunkNumber) {
    return new Path(intermediate, "probabilities/" + chunkNumber);
  }

  private void runForward() throws IOException, ClassNotFoundException, InterruptedException {
    logger.info("Running forward Viterbi pass");

    for (int i = 0; i < getChunkCount(); ++i) {
      logger.info("Processing chunk " + (i+1) + "/" + getChunkCount());
      Job job = new Job(getConf(), "viterbi-forward-" + i);
      job.setMapperClass(Mapper.class);
      job.setReducerClass(ForwardViterbiReducer.class);
      job.setInputFormatClass(SequenceFileInputFormat.class);
      job.setMapOutputKeyClass(Text.class);
      job.setMapOutputValueClass(ViterbiDataWritable.class);
      job.setOutputFormatClass(SequenceFileOutputFormat.class);
      job.setOutputKeyClass(Text.class);
      job.setOutputValueClass(ViterbiDataWritable.class);

      SequenceFileOutputFormat.setCompressOutput(job, true);
      SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK);

      job.setJarByClass(ForwardViterbiReducer.class);

      String chunk = ((Integer) i).toString();
      Path chunkInput = new Path(inputs, chunk);
      Path chunkIntermediate = getProbabilitiesPath(i);
      FileInputFormat.addInputPath(job, chunkInput);
      FileOutputFormat.setOutputPath(job, chunkIntermediate);
      if (i > 0) {
        //adding output of previous step
        FileInputFormat.addInputPath(job, getProbabilitiesPath(i-1));
      }

      Configuration jobConfiguration = job.getConfiguration();
      jobConfiguration.setInt("hmm.chunk_number", i);
      jobConfiguration.set("hmm.backpointers", intermediate + "/backpointers/" + chunk);
      String modelName = new Path(model).getName();

      jobConfiguration.set("hmm.model", modelName);
      DistributedCache.addCacheFile(URI.create(model), jobConfiguration);
      job.submit();
      job.waitForCompletion(true);
    }
  }

  public static void main(String[] args) throws Exception {
    System.exit(ToolRunner.run(new Configuration(), new ParallelViterbiDriver(), args));
  }
}

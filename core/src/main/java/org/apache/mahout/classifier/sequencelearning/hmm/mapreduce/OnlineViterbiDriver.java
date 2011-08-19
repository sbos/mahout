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
import org.apache.mahout.common.CommandLineUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;

public class OnlineViterbiDriver {
  private String inputs;
  private String output, intermediate, model;
  private Configuration configuration;

  private static Logger logger = LoggerFactory.getLogger(ParallelViterbiDriver.class);

  private OnlineViterbiDriver(String inputs, String output, String intermediate, String model) {
    this.inputs = inputs;
    this.output = output;
    this.intermediate = intermediate;
    this.model = model;

    configuration = new Configuration();
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
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

      String inputs = (String) commandLine.getValue(inputOption);
      String output = (String) commandLine.getValue(outputOption);
      String intermediate = (String) commandLine.getValue(intermediateOption);
      String modelPath = (String) commandLine.getValue(modelOption);

      OnlineViterbiDriver driver = new OnlineViterbiDriver(inputs, output, intermediate, modelPath);
      driver.runForward();
    } catch (OptionException e) {
      CommandLineUtil.printHelp(options);
    }
  }

  private int getChunkCount() throws IOException {
    FileSystem fs = FileSystem.get(URI.create(inputs), configuration);
    Path path = new Path(inputs);
    return fs.listStatus(path).length;
  }

  private Path getAlgorithmStatePath(int chunkNumber) {
    return new Path(intermediate, "state/" + chunkNumber);
  }

  private void runForward() throws IOException, ClassNotFoundException, InterruptedException {
    logger.info("Running forward Viterbi pass");

    for (int i = 0; i < getChunkCount(); ++i) {
      logger.info("Processing chunk " + (i+1) + "/" + getChunkCount());
      Job job = new Job(configuration, "online-viterbi-" + i);
      job.setMapperClass(Mapper.class);
      job.setReducerClass(OnlineViterbiReducer.class);
      job.setInputFormatClass(SequenceFileInputFormat.class);
      job.setMapOutputKeyClass(Text.class);
      job.setMapOutputValueClass(ViterbiDataWritable.class);
      job.setOutputFormatClass(SequenceFileOutputFormat.class);
      job.setOutputKeyClass(Text.class);
      job.setOutputValueClass(ViterbiDataWritable.class);

      SequenceFileOutputFormat.setCompressOutput(job, true);
      SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.RECORD);

      job.setJarByClass(OnlineViterbiReducer.class);

      String chunk = ((Integer) i).toString();
      Path chunkInput = new Path(inputs, chunk);
      Path chunkIntermediate = getAlgorithmStatePath(i);
      FileInputFormat.addInputPath(job, chunkInput);
      FileOutputFormat.setOutputPath(job, chunkIntermediate);
      if (i > 0) {
        //adding output of previous step
        FileInputFormat.addInputPath(job, getAlgorithmStatePath(i - 1));
      }

      Configuration jobConfiguration = job.getConfiguration();
      jobConfiguration.setInt("hmm.chunk_number", i);
      jobConfiguration.set("hmm.output", output);
      String modelName = new Path(model).getName();

      jobConfiguration.set("hmm.model", modelName);
      DistributedCache.addCacheFile(URI.create(model), jobConfiguration);
      job.submit();
      job.waitForCompletion(true);
    }
  }
}

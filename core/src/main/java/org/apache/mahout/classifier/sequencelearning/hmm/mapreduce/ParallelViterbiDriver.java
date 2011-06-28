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

public class ParallelViterbiDriver {
  private String inputs;
  private String output, intermediate, model;
  private Configuration configuration;

  private static Logger logger = LoggerFactory.getLogger(ParallelViterbiDriver.class);

  private ParallelViterbiDriver(String inputs, String output, String intermediate, String model) {
    this.inputs = inputs;
    this.output = output;
    this.intermediate = intermediate;
    this.model = model;

    configuration = new Configuration();
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    final DefaultOptionBuilder optionBuilder = new DefaultOptionBuilder();
    final ArgumentBuilder argumentBuilder = new ArgumentBuilder();

    final Option inputOption = optionBuilder.withLongName("input").
      withShortName("i").withRequired(true).
      withArgument(argumentBuilder.withMinimum(1).withMaximum(1).withName("path").create()).create();

    final Option outputOption = optionBuilder.withLongName("output").
      withShortName("o").withRequired(true).
      withArgument(argumentBuilder.withMinimum(1).withMaximum(1).
        withName("path").create()).create();

    final Option modelOption = optionBuilder.withLongName("model").
      withShortName("m").withRequired(true).
      withArgument(argumentBuilder.withMinimum(1).withMaximum(1).
        withName("path").create()).withDescription("Serialized HMM model").create();

    final Option intermediateOption = optionBuilder.withLongName("intermediate").
      withShortName("im").withRequired(true).
      withArgument(argumentBuilder.withMinimum(1).withMaximum(1).
        withName("path").withDefault("/tmp").create()).create();

    final Group options = new GroupBuilder().withOption(inputOption).
      withOption(outputOption).withOption(intermediateOption).
      withOption(modelOption).withName("Options").create();

    try {
      final Parser parser = new Parser();
      parser.setGroup(options);
      final CommandLine commandLine = parser.parse(args);

      final String inputs = (String) commandLine.getValue(inputOption);
      final String output = (String) commandLine.getValue(outputOption);
      final String intermediate = (String) commandLine.getValue(intermediateOption);
      final String modelPath = (String) commandLine.getValue(modelOption);

      final ParallelViterbiDriver driver = new ParallelViterbiDriver(inputs, output, intermediate, modelPath);
      driver.runForward();
      driver.runBackward();
    } catch (OptionException e) {
      CommandLineUtil.printHelp(options);
    }
  }

  private int getChunkCount() throws IOException {
    FileSystem fs = FileSystem.get(URI.create(inputs), configuration);
    Path path = new Path(inputs);
    return fs.listStatus(path).length;
  }

  private void runBackward() throws IOException, ClassNotFoundException, InterruptedException {
    logger.info("Running backward Viterbi pass");

    final int chunkCount = getChunkCount();
    for (int i = chunkCount; i >= 0; --i) {
      logger.info("Processing chunk " + (i+1) + "/" + chunkCount);
      final Job job = new Job(configuration, "viterbi-backward-" + i);
      job.setMapperClass(Mapper.class);
      job.setReducerClass(BackwardViterbiReducer.class);
      job.setInputFormatClass(SequenceFileInputFormat.class);
      job.setOutputKeyClass(SequenceKey.class);
      job.setOutputValueClass(BackwardViterbiData.class);
      job.setOutputFormatClass(SequenceFileOutputFormat.class);
      job.setMapOutputKeyClass(SequenceKey.class);
      job.setMapOutputValueClass(BackwardViterbiData.class);

      SequenceFileOutputFormat.setCompressOutput(job, true);
      SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK);

      job.setJarByClass(BackwardViterbiReducer.class);

      FileOutputFormat.setOutputPath(job, getLastStatePath(i));
      FileInputFormat.addInputPath(job, getBackpointersPath(i));
      if (i < chunkCount)
        FileInputFormat.addInputPath(job, getLastStatePath(i+1));
      FileInputFormat.addInputPath(job, getLastStatePathMR(i));
      final Configuration jobConfiguration = job.getConfiguration();
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

  private Path getLastStatePathMR(int chunkNumber) {
    return new Path(intermediate, "laststates-mr/" + chunkNumber);
  }

  private Path getProbabilitiesPath(int chunkNumber) {
    return new Path(intermediate, "probabilities/" + chunkNumber);
  }

  private void runForward() throws IOException, ClassNotFoundException, InterruptedException {
    logger.info("Running forward Viterbi pass");

    for (int i = 0; i < getChunkCount(); ++i) {
      logger.info("Processing chunk " + (i+1) + "/" + getChunkCount());
      final Job job = new Job(configuration, "viterbi-forward-" + i);
      job.setMapperClass(Mapper.class);
      job.setReducerClass(ForwardViterbiReducer.class);
      job.setInputFormatClass(SequenceFileInputFormat.class);
      job.setMapOutputKeyClass(SequenceKey.class);
      job.setMapOutputValueClass(ForwardViterbiData.class);
      job.setOutputFormatClass(SequenceFileOutputFormat.class);
      job.setOutputKeyClass(SequenceKey.class);
      job.setOutputValueClass(ForwardViterbiData.class);

      SequenceFileOutputFormat.setCompressOutput(job, true);
      SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK);

      job.setJarByClass(ForwardViterbiReducer.class);

      final String chunk = ((Integer) i).toString();
      final Path chunkInput = new Path(inputs, chunk);
      final Path chunkIntermediate = getProbabilitiesPath(i);
      if (chunkInput.getFileSystem(configuration).exists(chunkInput))
        FileInputFormat.addInputPath(job, chunkInput);
      FileOutputFormat.setOutputPath(job, chunkIntermediate);
      if (i > 0) {
        //adding output of previous step
        FileInputFormat.addInputPath(job, getProbabilitiesPath(i-1));
      }

      final Configuration jobConfiguration = job.getConfiguration();
      jobConfiguration.setInt("hmm.chunk_number", i);
      jobConfiguration.set("hmm.backpointers", intermediate + "/backpointers/" + chunk);
      jobConfiguration.set("hmm.laststates", intermediate + "/laststates-mr/" + chunk);
      final String modelName = new Path(model).getName();

      jobConfiguration.set("hmm.model", modelName);
      DistributedCache.addCacheFile(URI.create(model), jobConfiguration);
      job.submit();
      job.waitForCompletion(true);
    }
  }
}

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
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.CommandLineUtil;

import java.io.IOException;
import java.net.URI;

public class ParallelViterbiDriver {
  private String inputs;
  private String output, intermediate, model;
  private Configuration configuration;

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
    } catch (OptionException e) {
      CommandLineUtil.printHelp(options);
    }
  }

  private int getChunkCount() throws IOException {
    FileSystem fs = FileSystem.get(URI.create(inputs), configuration);
    Path path = new Path(inputs);
    return fs.listStatus(path).length;
  }

  private void runForward() throws IOException, ClassNotFoundException, InterruptedException {
    for (int i = 0; i < getChunkCount(); ++i) {
      final Job job = new Job(configuration, "viterbi-forward");
      job.setMapperClass(Mapper.class);
      job.setReducerClass(ForwardViterbiReducer.class);
      job.setInputFormatClass(SequenceFileInputFormat.class);
      job.setMapOutputKeyClass(SequenceKey.class);
      job.setMapOutputValueClass(GenericViterbiData.class);
      job.setOutputFormatClass(SequenceFileOutputFormat.class);
      job.setOutputKeyClass(SequenceKey.class);
      job.setOutputValueClass(GenericViterbiData.class);

      final String chunk = ((Integer) i).toString();
      FileInputFormat.addInputPath(job, new Path(inputs, chunk));
      //TODO: add output of previous step

      final Configuration jobConfiguration = job.getConfiguration();
      jobConfiguration.setInt("hmm.chunk_number", i);
      jobConfiguration.set("hmm.intermediate", intermediate);
      jobConfiguration.set("hmm.model", new Path(model).getName());
      DistributedCache.addCacheFile(URI.create(model), jobConfiguration);
      job.submit();

      break;
    }
  }
}

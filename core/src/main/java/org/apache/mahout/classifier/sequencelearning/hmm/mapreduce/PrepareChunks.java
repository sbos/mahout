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
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.CommandLineUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URI;
import java.util.*;

/**
 * A tool for dividing input sequences to chunks and merging them
 */
public final class PrepareChunks extends Configured implements Tool {
  private final static Logger log = LoggerFactory.getLogger(PrepareChunks.class);

  @Override
  public int run(String[] args) throws IOException {
    DefaultOptionBuilder optionBuilder = new DefaultOptionBuilder();
    ArgumentBuilder argumentBuilder = new ArgumentBuilder();

    Option chunkSizeOption = optionBuilder.withLongName("chunksize").withRequired(false).
      withDescription("a length (in observations number) of each chunk").withShortName("c").
      withArgument(argumentBuilder.withMaximum(1).withMinimum(1).withName("length").
        withDefault("65536").create()).
      create();

    Option unchunkOption = optionBuilder.withLongName("unchunk").withRequired(false).
      withDescription("Convert chunked output to raw file").withShortName("u").create();

    Option inputOption = optionBuilder.withLongName("input").withRequired(true).
      withDescription("directory containing observed sequences").withShortName("i").
      withArgument(argumentBuilder.withMinimum(1).withName("path").create()).create();

    Option outputOption = optionBuilder.withLongName("output").withRequired(true).
      withDescription("directory to write chunks").withShortName("o").
      withArgument(argumentBuilder.withMinimum(1).withMaximum(1).withName("path").create()).create();

    GroupBuilder groupBuilder = new GroupBuilder();
    Group group = groupBuilder.withName("Options").
      withOption(chunkSizeOption).withOption(inputOption).withOption(outputOption).
      withOption(unchunkOption).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine commandLine = parser.parse(args);

      Configuration configuration = getConf();

      if (commandLine.hasOption(unchunkOption)) {
        String input = (String) commandLine.getValue(inputOption);
        String output = (String) commandLine.getValue(outputOption);

        FileSystem inputFs = FileSystem.get(URI.create(input), configuration);
        FileSystem outputFs = FileSystem.get(URI.create(output), configuration);

        OutputStream outputStream = outputFs.create(new Path(output));
        PrintWriter writer = new PrintWriter(outputStream);

        int chunkNumber = 0;
        HiddenSequenceWritable decoded = new HiddenSequenceWritable();

        while (true) {
          Path chunkPath = new Path(input, String.valueOf(chunkNumber));
          if (!inputFs.exists(chunkPath))
            break;

          log.info("Reading " + input + ", chunk number " + chunkNumber);
          FileSystem fs = FileSystem.get(chunkPath.toUri(), configuration);
          SequenceFile.Reader reader = new SequenceFile.Reader(fs, chunkPath, configuration);

          IntWritable chunk = new IntWritable();
          while (reader.next(chunk)) {
            reader.getCurrentValue(decoded);

            for (int state: decoded.get()) {
              writer.print(state);
              writer.print(' ');
            }
          }

          ++chunkNumber;
          reader.close();
        }

        writer.close();
        outputStream.close();

      } else {
        int chunkSize = Integer.parseInt((String) commandLine.getValue(chunkSizeOption));
        List<String> inputs = commandLine.getValues(inputOption);
        String output = (String) commandLine.getValue(outputOption);

        //initializing output directory
        FileSystem outputFS = FileSystem.get(URI.create(output), configuration);
        Path outputPath = new Path(output);
        outputFS.mkdirs(outputPath);

        ChunkSplitter chunkSplitter = new ChunkSplitter(chunkSize, outputPath, configuration);

        //processing each input
        for (String input: inputs) {
          FileSystem inputFS = FileSystem.get(URI.create(input), configuration);
          Path inputPath = new Path(input);
          inputFS.listStatus(inputPath, chunkSplitter);
        }

        //chunkSplitter.close();
      }
    } catch (OptionException e) {
      CommandLineUtil.printHelp(group);
      return 1;
    }
    return 0;
  }

  static class ChunkSplitter implements PathFilter {
    int chunkSize = 64;
    Path outputPath;
    Configuration configuration;
    FileSystem outputFileSystem;
    Map<String, List<SequenceFile.Writer>> outputs = new HashMap<String, List<SequenceFile.Writer>>();
    //List<SequenceFile.Writer> outputs = new ArrayList<SequenceFile.Writer>();

    public ChunkSplitter(int chunkSize, Path outputPath, Configuration configuration) throws IOException {
      this.chunkSize = chunkSize;
      this.configuration = configuration;
      this.outputPath = outputPath;

      outputFileSystem = outputPath.getFileSystem(configuration);
    }

    public void process(Path inputPath, FileSystem fs) throws IOException {
      log.info("Splitting " + inputPath.getName() + " to chunks with size " + chunkSize);
      FSDataInputStream in =  fs.open(inputPath);
      String inputName = inputPath.getName();
      BufferedReader reader = new BufferedReader(new InputStreamReader(in));
      Scanner scanner = new Scanner(reader);

      for (int currentChunk = 0; ; ++currentChunk) {
        int[] chunkObservations = new int[chunkSize];
        int observationsRead;
        for (observationsRead = 0;
             observationsRead < chunkSize && scanner.hasNext(); ++observationsRead) {
          chunkObservations[observationsRead] = scanner.nextInt();
        }

        if (observationsRead > 0) {
          List<SequenceFile.Writer> chunkWriters = outputs.get(inputPath.toString());
          if (chunkWriters == null) {
            chunkWriters = new LinkedList<SequenceFile.Writer>();
            outputs.put(inputPath.toString(), chunkWriters);
          }
          if (chunkWriters.size() <= currentChunk) {
            log.debug("Opening new sequence file for chunk #" + currentChunk);
            Path chunkPath = new Path(outputPath, String.valueOf(currentChunk));
            if (!fs.exists(chunkPath)) fs.mkdirs(chunkPath);
            SequenceFile.Writer writer = SequenceFile.createWriter(outputFileSystem, configuration,
              new Path(chunkPath, inputPath.getName()),
              Text.class, ViterbiDataWritable.class, SequenceFile.CompressionType.RECORD);
            chunkWriters.add(writer);

            log.info("Splitting " + inputName + ", chunk #" + currentChunk);

            ObservedSequenceWritable chunk = new ObservedSequenceWritable(chunkObservations,
              observationsRead, currentChunk, !scanner.hasNextInt());

            log.info(observationsRead + " observations to write to this chunk");
            writer.append(new Text(inputPath.getName()),
              ViterbiDataWritable.fromObservedSequence(chunk));

            writer.close();
          }
        }

        if (observationsRead < chunkSize)
          break;
      }

      scanner.close();
      reader.close();
      in.close();
      log.info(inputName + " was splitted successfully");
    }

    @Override
    public boolean accept(Path path) {
      try {
        FileSystem fs = path.getFileSystem(configuration);
        if (fs.isFile(path))
          process(path, fs);
      } catch (IOException e) {
        e.printStackTrace();
      }
      return true;
    }

    /*public void close() throws IOException {
      for (List<SequenceFile.Writer> chunkWrites: outputs.values()) {
        for (SequenceFile.Writer)
      }
    }*/
  }

  public static void main(String[] args) throws Exception {
    System.exit(ToolRunner.run(new Configuration(), new PrepareChunks(), args));
  }
}

package org.apache.mahout.classifier.sequencelearning.hmm;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.util.Progressable;
import org.apache.mahout.common.CommandLineUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.List;
import java.util.Scanner;

/**
 * A tool for dividing input sequences to chunks
 */
public final class PrepareChunks {
    private final static Logger log = LoggerFactory.getLogger(PrepareChunks.class);

    public static void main(String[] args) throws IOException {
        final DefaultOptionBuilder optionBuilder = new DefaultOptionBuilder();
        final ArgumentBuilder argumentBuilder = new ArgumentBuilder();

        final Option chunkSizeOption = optionBuilder.withLongName("chunksize").withRequired(true).
                withDescription("a length (in observations number) of each chunk").withShortName("c").
                withArgument(argumentBuilder.withMaximum(1).withMinimum(1).withName("length").
                        withDefault("64").create()).
                create();

        final Option inputOption = optionBuilder.withLongName("input").withRequired(true).
                withDescription("directory contains observed sequences").withShortName("i").
                withArgument(argumentBuilder.withMinimum(1).withName("path").create()).create();

        final Option outputOption = optionBuilder.withLongName("output").withRequired(true).
                withDescription("directory to write chunks").withShortName("o").
                withArgument(argumentBuilder.withMinimum(1).withMaximum(1).withName("path").create()).create();

        final GroupBuilder groupBuilder = new GroupBuilder();
        final Group group = groupBuilder.withName("Options").
                withOption(chunkSizeOption).withOption(inputOption).withOption(outputOption).create();

        try {
            final Parser parser = new Parser();
            parser.setGroup(group);
            final CommandLine commandLine = parser.parse(args);

            final int chunkSize = Integer.parseInt((String) commandLine.getValue(chunkSizeOption));
            final List<String> inputs = commandLine.getValues(inputOption);
            final String output = (String) commandLine.getValue(outputOption);

            final Configuration configuration = new Configuration();

            //initializing output directory
            final FileSystem outputFS = FileSystem.get(URI.create(output), configuration);
            final Path outputPath = new Path(output);
            outputFS.mkdirs(outputPath);

            final ChunkSplitter chunkSplitter = new ChunkSplitter(chunkSize, outputPath, configuration);

            //processing each input
            for (String input: inputs) {
                final FileSystem inputFS = FileSystem.get(URI.create(input), configuration);
                final Path inputPath = new Path(input);
                inputFS.listStatus(inputPath, chunkSplitter);
            }
        } catch (OptionException e) {
            CommandLineUtil.printHelp(group);
        }
    }

    static class ChunkSplitter implements PathFilter {
        int chunkSize = 64;
        Path outputPath;
        Configuration configuration;
        FileSystem outputFileSystem;

        public ChunkSplitter(int chunkSize, Path outputPath, Configuration configuration) throws IOException {
            this.chunkSize = chunkSize;
            this.configuration = configuration;
            this.outputPath = outputPath;

            outputFileSystem = outputPath.getFileSystem(configuration);
        }

        public void process(Path inputPath, FileSystem fs) throws IOException {
            log.info("Splitting " + inputPath.getName() + " to chunks with size " + chunkSize);
            FSDataInputStream in =  fs.open(inputPath);
            final String inputName = inputPath.getName();
            //TODO: add different formatters
            final BufferedReader reader = new BufferedReader(new InputStreamReader(in));
            final Scanner scanner = new Scanner(reader);

            for (int currentChunk = 1; ; ++currentChunk) {
                log.info("Splitting " + inputName + ", chunk #" + currentChunk);
                final int[] chunkObservations = new int[chunkSize];
                int observationsRead;
                for (observationsRead = 0;
                     observationsRead < chunkSize && scanner.hasNext(); ++observationsRead) {
                    chunkObservations[observationsRead] = scanner.nextInt();
                }

                final Path chunkPath = new Path(outputPath, currentChunk + "/" + inputName);
                FSDataOutputStream out = outputFileSystem.create(chunkPath, new Progressable() {
                    @Override
                    public void progress() {
                        if (log.isInfoEnabled())
                            log.info("Writing chunk for " + chunkPath.getName());
                    }
                });

                for (int i = 0; i < observationsRead; ++i)
                    out.writeInt(chunkObservations[i]);

                log.info(observationsRead + " observations to write to this chunk");
                out.close();

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
                final FileSystem fs = path.getFileSystem(configuration);
                if (fs.isFile(path))
                    process(path, fs);
            } catch (IOException e) {
                e.printStackTrace();
            }
            return true;
        }
    }
}

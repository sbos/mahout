package org.apache.mahout.classifier.sequencelearning.hmm;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.util.Tool;

/**
 * A tool for dividing input sequences to chunks
 */
public class PrepareChunks extends Configured implements Tool {
    static {
        Configuration.addDefaultResource("hdfs-default.xml");
        Configuration.addDefaultResource("hdfs-site.xml");
    }

    @Override
    public int run(String[] args) throws Exception {
        final Configuration conf = getConf();
        final int chunkSize = Integer.parseInt(conf.get("hmm.chunksize", "2"));
        System.out.println(chunkSize);
        return 0;
    }
}

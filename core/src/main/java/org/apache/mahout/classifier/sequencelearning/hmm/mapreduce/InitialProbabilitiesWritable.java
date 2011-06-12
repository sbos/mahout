package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;

/**
 */
class InitialProbabilitiesWritable extends ArrayWritable {
    public InitialProbabilitiesWritable() {
        super(DoubleWritable.class);
    }
}

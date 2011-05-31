package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 *
 * @author Sergey Bartunov
 */
public class ObservedSequenceWritable implements Writable {
    int[] data;

    public ObservedSequenceWritable() {

    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        //To change body of implemented methods use File | Settings | File Templates.
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
    }
}

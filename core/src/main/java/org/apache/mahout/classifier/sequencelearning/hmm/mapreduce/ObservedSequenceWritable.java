package org.apache.mahout.classifier.sequencelearning.hmm.mapreduce;

import org.apache.commons.lang.NullArgumentException;
import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

/**
 * The class for modeling a sequence of observed variable for parallel functionality in MapReduce
 */
public class ObservedSequenceWritable implements WritableComparable<ObservedSequenceWritable>, Cloneable {
    int[] data;

    public ObservedSequenceWritable(int length) {
        data = new int[length];
    }

    public ObservedSequenceWritable(int[] data) {
        this.data = new int[data.length];
        System.arraycopy(data, 0, this.data, 0, data.length);
    }

    @Override
    public ObservedSequenceWritable clone() {
        return new ObservedSequenceWritable(data);
    }

    public void assign(int value) {
        Arrays.fill(data, value);
    }

    public void assign(int[] values) {
        final int length = Math.min(getLength(), values.length);
        for (int i = 0; i < length; ++i)
            data[i] = values[i];
    }

    public int[] getData() {
        return data;
    }

    public void setData(int[] data) {
        if (data == null)
            throw new NullArgumentException("data");
        this.data = data;
    }

    public int getLength() {
        return data.length;
    }

    public void setLength(int length) {
        int[] newData = new int[length];
        if (data == null) {
            data = newData;
            return;
        }
        System.arraycopy(data, 0, newData, 0, Math.min(length, data.length));
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        dataOutput.writeInt(data.length);
        for (int i = 0; i < data.length; ++i)
            dataOutput.writeInt(data[i]);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        final int length = dataInput.readInt();
        final int[] data = new int[length];
        for (int i = 0; i < length; ++i)
            data[i] = dataInput.readInt();
    }

    @Override
    public int compareTo(ObservedSequenceWritable observedSequenceWritable) {
        final int lengthDifference = getLength() - observedSequenceWritable.getLength();
        if (lengthDifference != 0)
            return lengthDifference;
        final int[] otherData = observedSequenceWritable.getData();
        for (int i = 0; i < getLength(); ++i) {
            final int difference = data[i] - otherData[i];
            if (difference != 0)
                return difference;
        }
        return 0;
    }

    @Override
    public boolean equals(Object other) {
        return (other instanceof ObservedSequenceWritable) ? (compareTo((ObservedSequenceWritable)other) == 0) : false;
    }

    @Override
    public int hashCode() {
        int hash = ((Integer)data.length).hashCode();
        for (int i = 0; i < data.length; ++i)
            hash += i * ((Integer)data[i]).hashCode();
        return hash;
    }
}

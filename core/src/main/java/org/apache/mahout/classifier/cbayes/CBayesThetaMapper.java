package org.apache.mahout.classifier.cbayes;

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

import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.GenericsUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;


public class CBayesThetaMapper extends MapReduceBase implements
    Mapper<Text, FloatWritable, Text, FloatWritable> {

  private static final Logger log = LoggerFactory.getLogger(CBayesThetaMapper.class);   

  public HashMap<String, Float> labelWeightSum = null;
  String labelWeightSumString = " ";
  Float sigma_jSigma_k = 0.0f;
  String sigma_jSigma_kString = " ";
  Float vocabCount = 0.0f;
  String vocabCountString = " ";
  
  /**
   * We need to calculate the idf of each feature in each label
   * 
   * @param key The label,feature pair (can either be the freq Count or the term
   *        Document count
   * @param value
   * @param output
   * @param reporter
   * @throws IOException
   */
  public void map(Text key, FloatWritable value,
      OutputCollector<Text, FloatWritable> output, Reporter reporter)
      throws IOException {

    String labelFeaturePair = key.toString();
    float alpha_i = 1.0f;
    
    if (labelFeaturePair.startsWith(",")) { // if it is from the Sigma_j folder
                                            // (feature weight Sum)
      String feature = labelFeaturePair.substring(1);
      for (String label : labelWeightSum.keySet()) {
        double inverseDenominator = 1.0 /(sigma_jSigma_k - labelWeightSum.get(label) + vocabCount);
        FloatWritable weight = new FloatWritable((float)((value.get() + alpha_i)*inverseDenominator ));
        output.collect(new Text((label + "," + feature).trim()), weight); //output Sigma_j
      }
    } else {
      String label = labelFeaturePair.split(",")[0];
      double inverseDenominator = 1.0 /(sigma_jSigma_k - labelWeightSum.get(label) + vocabCount);
      FloatWritable weight = new FloatWritable((float)(-1 * value.get()  * inverseDenominator));
      output.collect(key, weight);//output -D_ij       
    }
  }

  @Override
  public void configure(JobConf job) {
    try {
      if (labelWeightSum == null) {
        labelWeightSum = new HashMap<String, Float>();

        DefaultStringifier<HashMap<String, Float>> mapStringifier = new DefaultStringifier<HashMap<String, Float>>(
            job, GenericsUtil.getClass(labelWeightSum));

        labelWeightSumString = mapStringifier.toString(labelWeightSum);
        labelWeightSumString = job.get("cnaivebayes.sigma_k",
            labelWeightSumString);
        labelWeightSum = mapStringifier.fromString(labelWeightSumString);

        DefaultStringifier<Float> floatStringifier = new DefaultStringifier<Float>(
            job, GenericsUtil.getClass(sigma_jSigma_k));
        sigma_jSigma_kString = floatStringifier.toString(sigma_jSigma_k);
        sigma_jSigma_kString = job.get("cnaivebayes.sigma_jSigma_k",
            sigma_jSigma_kString);
        sigma_jSigma_k = floatStringifier.fromString(sigma_jSigma_kString);
        
        vocabCountString = floatStringifier.toString(vocabCount);
        vocabCountString = job.get("cnaivebayes.vocabCount",
            vocabCountString);
        vocabCount = floatStringifier.fromString(vocabCountString);
       
      }
    } catch (IOException ex) {
      log.info(ex.toString(), ex);
    }
  }
}
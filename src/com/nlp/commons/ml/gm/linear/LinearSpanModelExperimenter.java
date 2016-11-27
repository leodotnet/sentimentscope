/** Statistical Natural Language Processing System
    Copyright (C) 2014  Lu, Wei

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.nlp.commons.ml.gm.linear;

import java.io.IOException;

import com.nlp.commons.types.Instance;
import com.nlp.hybridnetworks.DiscriminativeNetworkModel;
import com.nlp.hybridnetworks.GenerativeNetworkModel;
import com.nlp.hybridnetworks.NetworkConfig;
import com.nlp.hybridnetworks.GlobalNetworkParam;
import com.nlp.hybridnetworks.NetworkModel;

public class LinearSpanModelExperimenter {
	
	public static void main(String args[]) throws IOException, InterruptedException{
		
		String train_filename = "data/ctb5/ctb5.train";
		String test_filename = "data/ctb5/ctb5.test";
		
		NetworkConfig.L2_REGULARIZATION_CONSTANT = 0.0;
		
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = false;
		NetworkConfig._numThreads = 10;//Integer.parseInt(args[0]);
		int numInstances = 40;//Integer.parseInt(args[1]);
		
		LinearSpanInstance[] train_instances = LinearSpanInstanceReader.readTrain_column(train_filename, numInstances);
		
		OutputTag[] outputTags = LinearSpanInstanceReader.getOutputTags();
		
		LinearSpanFeatureManager fm = new LinearSpanFeatureManager(new GlobalNetworkParam(), outputTags);
		
		LinearSpanNetworkCompiler compiler = new LinearSpanNetworkCompiler(outputTags);
		
		NetworkModel model = NetworkConfig.TRAIN_MODE_IS_GENERATIVE ? GenerativeNetworkModel.create(fm, compiler)
				: DiscriminativeNetworkModel.create(fm, compiler);
		
		model.train(train_instances, 1000);
		
		train_instances = LinearSpanInstanceReader.readTest_column(train_filename);
		LinearSpanInstance[] test_instances = LinearSpanInstanceReader.readTest_column(test_filename);
		
		double num_outputs;
		double num_predictions;
		double num_correct;
		
		for(LinearSpanInstance[] instances : new LinearSpanInstance[][]{train_instances, test_instances}){
			if(instances == train_instances)
				System.err.println("=TRAINING SET=");
			if(instances == test_instances)
				System.err.println("=EVALUATION SET=");
			
			num_outputs = 0;
			num_predictions = 0;
			num_correct = 0;
			
			Instance[] predicted_instances = model.decode(instances);
			
			for(int k = 0; k<predicted_instances.length; k++){
				LinearSpanInstance output = (LinearSpanInstance)predicted_instances[k];
				num_outputs += output.countNumOutputs();
				num_predictions += output.countNumPredictions();
				num_correct += output.countNumCorrectlyPredicted();
			}
			System.err.println("#OUTP="+num_outputs);
			System.err.println("#PRED="+num_predictions);
			System.err.println("#CORR="+num_correct);
			
			double P = num_correct/num_predictions;
			double R = num_correct/num_outputs;
			double F = 2 / (1/P + 1/R);
			
			System.err.println("P.="+P);
			System.err.println("R.="+R);
			System.err.println("F.="+F);
		}
		
	}
	
}
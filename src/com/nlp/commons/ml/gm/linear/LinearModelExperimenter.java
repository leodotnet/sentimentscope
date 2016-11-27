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

public class LinearModelExperimenter {
	
	public static void main(String args[]) throws IOException, InterruptedException{
		
		String folder = "twitter/project";
		String train_filename = "data/linear/"+folder+"/train";
		String test_filename = "data/linear/"+folder+"/test/test.out";
		
//		String train_filename = "data/ctb5/ctb5.train.1";
//		String test_filename = "data/ctb5/ctb5.test";
		
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = true;
		
		LinearInstance[] train_instances = LinearInstanceReader.readTrain_column(train_filename);
		
		OutputTag[] outputTags = LinearInstanceReader.getOutputTags();
		
		LinearFeatureManager fm = new LinearFeatureManager(new GlobalNetworkParam(), outputTags);
		
		LinearNetworkCompiler compiler = new LinearNetworkCompiler(outputTags);
		
		NetworkModel model = NetworkConfig.TRAIN_MODE_IS_GENERATIVE ? GenerativeNetworkModel.create(fm, compiler)
				: DiscriminativeNetworkModel.create(fm, compiler);
		
		model.train(train_instances, 100);
		
		train_instances = LinearInstanceReader.readTest_column(train_filename);
		LinearInstance[] test_instances = LinearInstanceReader.readTest_column(test_filename);
		
		int num_total_tags;
		int num_corr_tags;
		
//		for(LinearInstance[] instances : new LinearInstance[][]{train_instances}){
		for(LinearInstance[] instances : new LinearInstance[][]{train_instances, test_instances}){
			if(instances == train_instances)
				System.err.println("=TRAINING SET=");
			if(instances == test_instances)
				System.err.println("=EVALUATION SET=");
			
			num_total_tags = 0;
			num_corr_tags = 0;
			
			Instance[] outputs = model.decode(instances);
			
			for(int k = 0; k<outputs.length; k++){
				LinearInstance output = (LinearInstance)outputs[k];
				num_total_tags += output.size();
				num_corr_tags += output.countNumCorrectlyPredicted();
			}
			System.err.println("#TOTL="+num_total_tags);
			System.err.println("#CORR="+num_corr_tags);
			System.err.println("ACCUR="+(double)num_corr_tags/(double)num_total_tags);
		}
		
	}
	
}
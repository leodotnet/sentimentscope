package com.nlp.targetedsentiment.f.baseline.joint;

import java.io.IOException;

import com.nlp.hybridnetworks.NetworkConfig;
import com.nlp.targetedsentiment.util.TargetSentimentGlobal;


public class PipeTargetSentimentLearner {
	
	public static String in_path = "data//Twitter_";
	public static String out_path = "experiments//sentiment//model//baseline_joint//Twitter_";
	public static String feature_file_path = in_path + "//feature_files//";
	public static boolean visual = false;
	public static String lang = "en";
	public static boolean weightpush = false;
	public static boolean word_feature_on = true;
	public static String subpath = "default";

	public static void main(String[] args) throws InterruptedException, IOException, ClassNotFoundException {
		
		

		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = false;
		NetworkConfig.L2_REGULARIZATION_CONSTANT = 0.0005;
		NetworkConfig.RANDOM_INIT_WEIGHT = false;
		int num_iter = 1000;
		
		int begin_index = 2;
		int end_index = 10;
		
		//#iter, #L2, #begin_index, end_index
		if (args.length > 0)
		{
			num_iter = Integer.parseInt(args[0]);
			NetworkConfig.L2_REGULARIZATION_CONSTANT = Double.parseDouble(args[1]);
		}
		if (args.length > 2)
		{
			begin_index = Integer.parseInt(args[2]);
			end_index = Integer.parseInt(args[3]);
		}
		
		if (args.length > 4)
		{
			if (!args[4].trim().equals(""))
				lang = args[4];
		}
		
		if (args.length > 5)
		{
			if (args[5].trim().equals("weightpush"))
			{
				weightpush = true;
			}
		}
		
		if (args.length > 6)
		{
			if (args[6].length() > 0)
				subpath = args[6];
		}
		
		in_path = in_path + lang + "//";
		out_path = out_path + lang + "//" + subpath + "//";
		feature_file_path = in_path + "//feature_files//";
		
		if (lang.equals("es"))
		{
			TargetSentimentGlobal.LinguisticFeaturesLibaryName = TargetSentimentGlobal.LinguisticFeaturesLibaryName_es;
			TargetSentimentGlobal.LinguisticFeaturesLibaryNamePart = TargetSentimentGlobal.LinguisticFeaturesLibaryNamePart_es;
		}
		
		
		System.out.println("#iter=" + num_iter + " L2=" + NetworkConfig.L2_REGULARIZATION_CONSTANT + " lang=" + lang );
	
		if (visual)
		{
			// Visualize Demo
			visualize();
			return;
		}
		
		String train_file;
		String test_file;
		String model_file;
		String result_file;
		String iter = num_iter + "";
		String weight_push;
		String paper_features;
		
		for(int i = begin_index; i <= end_index; i++)
		{
			System.out.println("Executing Data " + i);
			train_file = in_path + "train." + i +".coll";
			test_file = in_path + "test." + i + ".coll";
			model_file = out_path + "joint." + i + ".model";
			result_file = out_path + "result." + i + ".out";
			weight_push = in_path + "weight0.data";
			//paper_features = out_path + "paperfeatures//" + "pipeline_sent.with_lex_id." + i + ".volitional-best.ff";
		
			PipeTargetSentimentAlgoModel algomodel = new PipeTargetSentimentAlgoModel();
		
			if (!weightpush)
			{
				args = new String[]{train_file, model_file, iter};
				algomodel.setDemoMode(true);
				algomodel.Train(args);
				algomodel.setDemoMode(true);
				//algomodel.loadWeightfromFile(paper_features);
			}
			args = new String[]{test_file, model_file, result_file, test_file, weight_push};
			
			//algomodel.Evaluate(args);
		}
	}

	
	public static void train()
	{
		

	}
	
	
	public static void evaluate() {
		
	
		
	}
	
	public static void visualize()
	{
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = false;
		NetworkConfig.L2_REGULARIZATION_CONSTANT = 0.5;
		int num_iter = 2000;
		
		int begin_index = 1;
		int end_index = begin_index;
		
		/* Visualize Demo
		algomodel.setDemoMode(true);
		algomodel.Train(args);
		algomodel.Visualize(new String[]{"2"});
		return;
		*/
		String train_file;
		String test_file;
		String model_file;
		String result_file;
		String iter = num_iter + "";
		int i = 1;
		PipeTargetSentimentAlgoModel algomodel = new PipeTargetSentimentAlgoModel();
		train_file = in_path + "train." + i +".coll";
		test_file = in_path + "test." + i + ".coll";
		model_file = out_path + "3node." + i + ".model";
		result_file = out_path + "result." + i + ".out";
		String weight_push = in_path + "weight0.data";
		
		/* Visualize Demo*/
		algomodel.setDemoMode(true);
		String[] args = new String[]{train_file, model_file, iter};
		algomodel.Train(args);
		algomodel.Visualize(new String[]{"1"});
		
		
	}
}

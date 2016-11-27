package com.nlp.targetedsentiment.util;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.nlp.algomodel.AlgoGlobal;
import com.nlp.algomodel.AlgoModel;
import com.nlp.hybridnetworks.FeatureArray;
import com.nlp.hybridnetworks.NetworkConfig;






public class TargetSentimentGlobal {
	public static boolean WITHOUT_HIDDEN_SENTIMENT = false;
	public static boolean USE_POS_TAG = false;
	public static boolean ENABLE_WORD_EMBEDDING = false;
	public static int NEMaxLength = 6;
	public static boolean WEIGHT_PUSH = false;
	public static boolean SAVE_FEATURE = false;
	public static boolean ECHO_FEATURE = false;
	public static boolean DUMP_FEATURE = true;
	public static boolean DEBUG = false;
	public static boolean OUTPUT_SENTIMENT_SPAN = false;
	public static ArrayList<Integer> SENTIMENT_SPAN_SPLIT = new ArrayList<Integer>();
	public static boolean ENABLE_CRFNN_FEATURE = false;
	
	
	public static WordEmbedding Word2Vec = null;
	public static void initWordEmbedding(String lang)
	{
		FeatureArray.ENABLE_WORD_EMBEDDING = true;
		Word2Vec = new WordEmbedding(lang);
	}
	
	public static String[] LinguisticFeaturesLibaryName = new String[] {
		"bad_words", "good_words", "curse_words", "prepositions",
		"determiners", "syllables", "Intensifiers_Eng.txt", "Neg_Abbrev_Eng.txt",
		"Neg_Prefix_Eng.txt", "Neg_Slang_Eng.txt", "Pos_Abbrev_Eng.txt",
		"Pos_Slang_Eng.txt" };

	public static String[] LinguisticFeaturesLibaryName_es = new String[] {
		"bad_words", "good_words", "curse_words", "prepositions",
		"determiners", "syllables","Intensifiers_Sp.txt", "Neg_Abbrev_Sp.txt",
		"Neg_Prefix_Sp.txt", "Neg_Slang_Sp.txt", "Pos_Abbrev_Sp.txt",
		"Pos_Slang_Sp.txt" };
	
	public static String[] LinguisticFeaturesLibaryNamePart = new String[]{"Neg_Prefix_Eng.txt"};
	
	public static String[] LinguisticFeaturesLibaryNamePart_es = new String[]{"Neg_Prefix_Sp.txt"};
	
	
	public static String in_path = "data//Twitter_";
	public static String out_path = "experiments//sentiment//<modelname>//Twitter_";
	public static String feature_file_path = in_path + "//feature_files//";
	public static boolean visual = false;
	public static String lang = "en";
	public static boolean word_feature_on = true;
	public static String subpath = "default";
	public static String modelname = "sentimentspan_latent";
	
	public static void main(String[] args) throws InstantiationException, IllegalAccessException, IOException
	{
		//NetworkConfig.ENABLE_MAX_MARGINAL = true;
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = false;
		NetworkConfig.L2_REGULARIZATION_CONSTANT = 0.0005;
		NetworkConfig.RANDOM_INIT_WEIGHT = false;
		NetworkConfig._numThreads = 20;
		TargetSentimentGlobal.WITHOUT_HIDDEN_SENTIMENT = false;
		TargetSentimentGlobal.USE_POS_TAG = false;
		TargetSentimentGlobal.ENABLE_WORD_EMBEDDING = false;
		TargetSentimentGlobal.NEMaxLength = 6;

		
		int num_iter = 1000;
		
		int begin_index = 2;
		int end_index = 10;
		int pArg = 0;
		
		System.out.println(Arrays.toString(args));
		
		if (args.length > pArg)
		{
			modelname = args[pArg++];
		}
		
		//#iter, #L2, #begin_index, end_index
		if (args.length > pArg)
		{
			num_iter = Integer.parseInt(args[pArg++]);
			NetworkConfig.L2_REGULARIZATION_CONSTANT = Double.parseDouble(args[pArg++]);
		}
		if (args.length > pArg + 1)
		{
			begin_index = Integer.parseInt(args[pArg++]);
			end_index = Integer.parseInt(args[pArg++]);
		}
		
		if (args.length > pArg)
		{
			String t = args[pArg++];
			if (!t.trim().equals(""))
				lang = t;
		}
		
		if (args.length > pArg)
		{
			if (args[pArg++].trim().equals("weightpush"))
			{
				WEIGHT_PUSH = true;
			}
		}
		
		if (args.length > pArg)
		{
			String t = args[pArg++];
			if (t.length() > 0)
				subpath = t;
		}
		
		if (args.length > pArg)
		{
			String t = args[pArg++];
			if (t.length() > 0)
				TargetSentimentGlobal.NEMaxLength = Integer.parseInt(t);
			
		}
		
		parseArgument(args, pArg);
		
		
		
		in_path = in_path + lang + "//";
		
		if (modelname.startsWith("baseline_pipeline"))
		{
			out_path = out_path.replace("<modelname>", "baseline_pipeline");
		}
		else
		{
			out_path = out_path.replace("<modelname>", modelname);
			
		}
		out_path = out_path + lang + "//" + subpath + "//";
		feature_file_path = in_path + "//feature_files//";
		
		if (modelname.equals("baseline_pipelineSent"))
		{
			in_path = "experiments//sentiment//model//baseline_pipeline//Twitter_" + lang + "//temp//";
		}
		
		if (lang.equals("es"))
		{
			LinguisticFeaturesLibaryName = LinguisticFeaturesLibaryName_es;
			LinguisticFeaturesLibaryNamePart = LinguisticFeaturesLibaryNamePart_es;
		}
		
		System.out.println("ENABLE_WORD_EMBEDDING=" + TargetSentimentGlobal.ENABLE_WORD_EMBEDDING);
		if (TargetSentimentGlobal.ENABLE_WORD_EMBEDDING)
		{
			TargetSentimentGlobal.initWordEmbedding(lang);
		}
		
		
		System.out.println("#iter=" + num_iter + " L2=" + NetworkConfig.L2_REGULARIZATION_CONSTANT + " lang=" + lang );
	
		
		String train_file;
		String test_file;
		String model_file;
		String result_file;
		String iter = num_iter + "";
		String weight_push;
		
		for(int i = begin_index; i <= end_index; i++)
		{
			System.out.println("Executing Data " + i);
			train_file = in_path + "train." + i +".coll";
			test_file = in_path + "test." + i + ".coll";
			model_file = out_path + "3node." + i + ".model";
			result_file = out_path + "result." + i + ".out";
			weight_push = in_path + "weight0.data";
			
			if (modelname.equals("baseline_pipelineNE"))
			{
				train_file = in_path + "train." + i +".coll";
				test_file = in_path + "test." + i + ".coll";
				model_file = out_path + "3node." + i + ".model";
				result_file = out_path + "result." + i + ".NE.out";
			}
			else if (modelname.equals("baseline_pipelineSent"))
			{
				train_file = in_path + "train." + i +".pipeline.sent";
				test_file = in_path + "test." + i + ".pipeline.sent";
				model_file = out_path + "pipelineSent." + i + ".model";
				result_file = out_path + "result." + i + ".out";
			}
		
			Class model = AlgoGlobal.parseClass(modelname);
			AlgoModel algomodel = (AlgoModel)(model.newInstance());
		
			if (!WEIGHT_PUSH)
			{
				NetworkConfig._numThreads = 20;
				args = new String[]{train_file, model_file, iter};
				algomodel.Train(args);
				algomodel.setDemoMode(true);
			}
			args = new String[]{test_file, model_file, result_file, test_file, weight_push};
			NetworkConfig._numThreads = 1;
			algomodel.Evaluate(args);
			
			/*
			if (modelname.equals("baseline_pipelineNE"))
			{
				System.out.println("Preparing data for 2nd stage of pipeline...");
				String[] parameters = new String[]{"python","scripts//sentiment_pipeline.py", begin_index + "", end_index + "", lang, subpath};
				ProcessBuilder pb = new ProcessBuilder(parameters);
				pb.redirectErrorStream(true);
				final Process p = pb.start();
				System.out.println("Done");
			}*/
		}
	}
	
	public static void parseArgument(String[] args, int pArgs)
	{
		
		
		System.out.print("Pasring additional arguments:");
		for(int i = pArgs; i < args.length; i++)
		{
			String t = args[i];
			System.out.print(t + "\t");
			
			if (t.equals("SAVE_FEATURE"))
			{
				SAVE_FEATURE = true;
			}
			else if (t.equals("DUMP_FEATURE"))
			{
				DUMP_FEATURE = true;
			}
			else if (t.equals("ECHO_FEATURE"))
			{
				ECHO_FEATURE = true;
			}
			else if (t.equals("WITHOUT_HIDDEN_SENTIMENT"))
			{
				WITHOUT_HIDDEN_SENTIMENT = true;
			}
			else if (t.equals("USE_POS_TAG"))
			{
				USE_POS_TAG = true;
			}
			else if (t.equals("ENABLE_WORD_EMBEDDING"))
			{
				ENABLE_WORD_EMBEDDING = true;
			}
			else if (t.equals("DEBUG"))
			{
				DEBUG = true;
			}
			else if (t.equals("WORD_FEATURE_OFF"))
			{
				word_feature_on = false;
				System.out.println("Turn off word features");
			}
		}
		System.out.println();
	}
	
	

}

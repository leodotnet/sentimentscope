package com.nlp.targetedsentiment.f.baseline.joint;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Scanner;
import java.util.Set;

import com.nlp.algomodel.AlgoModel;
import com.nlp.algomodel.parser.VerticalColumnInstanceParser;
import com.nlp.commons.types.InputToken;
import com.nlp.commons.types.Instance;
import com.nlp.commons.types.OutputToken;
import com.nlp.hybridnetworks.DiscriminativeNetworkModel;
import com.nlp.hybridnetworks.GenerativeNetworkModel;
import com.nlp.hybridnetworks.GlobalNetworkParam;
import com.nlp.hybridnetworks.LocalNetworkLearnerThread;
import com.nlp.hybridnetworks.NetworkConfig;
import com.nlp.hybridnetworks.TableLookupNetwork;
import com.nlp.targetedsentiment.f.baseline.PipeTargetSentimentFeatureManager.*;
import com.nlp.targetedsentiment.util.TargetSentimentGlobal;


public class PipeTargetSentimentAlgoModel extends AlgoModel {
	
	private static final boolean ECHO_FEATURE = false;

	VerticalColumnInstanceParser p;
	
	public static String MODELNAME = "TargetSentimentAlgoModel";
	
	static boolean SIMILAR_TAG = false;
	
	public static int[][][] inst_features = null;
	
	PipeTargetSentimentFeatureManager myfm;
	
 
	public PipeTargetSentimentAlgoModel() {
		super();
		this.demoMode = true;
	}
	
	@Override
	protected void initInstanceParser(AlgoModel algomodel) {
		p = new VerticalColumnInstanceParser(algomodel);
		
		this.parser = p;
	}
	
	
	@Override
	protected void initFeatureManager() {
		
		myfm = new PipeTargetSentimentFeatureManager(param);
		this.fm = myfm;
		myfm.loadLinguisticFeatureLibary();
	}
	
	
	
	@Override
	protected void initNetworkCompiler()
	{
		this.compiler = new PipeTargetSentimentCompiler();
		//((PipeTargetSentimentCompiler)this.compiler).setFeatureManager((PipeTargetSentimentCompiler)this.fm);
	}


	@Override
	protected void saveModel() throws IOException {
		
		String filename_model = (String) getParameters("filename_model");
		
		System.out.println();
		System.err.println("Saving Model...");
		ObjectOutputStream out;
		out = new ObjectOutputStream(new FileOutputStream(filename_model));
		out.writeObject(fm.getParam_G());
		out.flush();
		out.close();
		System.err.println("Model Saved.");
		
		if (TargetSentimentGlobal.DUMP_FEATURE)
			printFeature();
	}
	
	void printFeature()
	{
		
		GlobalNetworkParam paramG = fm.getParam_G();
		String filename_model = (String) getParameters("filename_model");
		
		PrintWriter modelTextWriter = null;
		try {
			modelTextWriter = new PrintWriter(filename_model + ".dump");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


		modelTextWriter.println("Num features: "+paramG.countFeatures());

		modelTextWriter.println("Features:");

		HashMap<String, HashMap<String, HashMap<String, Integer>>> featureIntMap = paramG.getFeatureIntMap();
		
		//System.out.println(featureIntMap.keySet());

		for(String featureType: sorted(featureIntMap.keySet())){

		     //.println(featureType);

		     HashMap<String, HashMap<String, Integer>> outputInputMap = featureIntMap.get(featureType);

		     for(String output: sorted(outputInputMap.keySet())){

		          //modelTextWriter.println("\t"+output);

		          HashMap<String, Integer> inputMap = outputInputMap.get(output);

		          for(String input: inputMap.keySet()){

		               int featureId = inputMap.get(input);

		               modelTextWriter.println(featureType + input+ ":= " + output + "="+fm.getParam_G().getWeight(featureId));
		               if (ECHO_FEATURE)
		            	   System.out.println(featureType + input+ ":= " + output + "="+fm.getParam_G().getWeight(featureId));
		          
		          }

		     }
		     
		     modelTextWriter.flush();

		}

		modelTextWriter.close();
	}
	
	ArrayList<String> sorted(Set set)
	{
		ArrayList<String> list = new ArrayList<String>(set);     
		Collections.sort(list);
		return list;
	}

	@Override
	protected void loadModel() throws IOException {
		
		
		String filename_model = (String) getParameters("filename_model");
		
		System.err.println("Loading Model...");
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename_model));
		try {
			param = (GlobalNetworkParam)in.readObject();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		in.close();
			
		
		System.err.println("Model Loaded.");
		
		String filename_weightpush = (String) getParameters("filename_weightpush");
		
		if(TargetSentimentGlobal.WEIGHT_PUSH)
		{	
			Scanner scanner = new Scanner(new File(filename_weightpush));
			int f1 = param.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_Start_Feature", "", "");
			double weightNE1 = param.getWeight(f1);
			double weightNE0 = 0;
			
			String t = scanner.nextLine().trim();
			if (!t.startsWith("NO"))
			{
				weightNE0 = Double.parseDouble(t);
				param.setWeight(f1, weightNE0);
				System.err.println("NE_Start_Feature="+param.getWeight(f1) + "\t old value=" + weightNE1);
			}
			
			
			int f2 = param.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_sentiment_Feature", "", "");
			double weightSENT1 = param.getWeight(f2);
			double weightSENT0 = 0;
			t = scanner.nextLine().trim();
			if (!t.startsWith("NO"))
			{
				weightSENT0 = Double.parseDouble(t);
				param.setWeight(f2, weightSENT0);
				System.err.println("NE_sentiment_Feature="+param.getWeight(f2) + "\t old value=" + weightSENT1);
			}
			
			scanner.close();
		}
		
	}
	
	
	

	@Override
	public void initTraining(String[] args) {
		setParameters("filename_input", args[0]);
		setParameters("filename_model", args[1]);
		this.MaxIteration = Integer.parseInt(args[2]);
	}

	@Override
	public void initEvaluation(String[] args) {
		setParameters("filename_input", args[0]);
		setParameters("filename_model", args[1]);
		setParameters("filename_output", args[2]);
		setParameters("filename_standard", args[3]);
		setParameters("filename_weightpush", args[4]);
	}

	@Override
	protected void evaluationResult(Instance[] instances_outputs) {
		String filename_output = (String) getParameters("filename_output");
		String filename_standard =  (String) getParameters("filename_standard");
		
		PrintWriter p = null;
		try {
			p = new PrintWriter(new File(filename_output));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		if (TargetSentimentGlobal.DEBUG)
			System.out.println("POS Tagging Result: ");
		for(int i = 0; i < instances_outputs.length; i++)
		{
			if (TargetSentimentGlobal.DEBUG)
				System.out.println("Testing case #" + i + ":");
			
			InputToken[] input = (InputToken[])instances_outputs[i].getInput();
			OutputToken[] output = (OutputToken[])instances_outputs[i].getPrediction();
			
			if (TargetSentimentGlobal.DEBUG)
			{
				System.out.println(Arrays.toString(input));
				System.out.println(Arrays.toString(output));
			}
			for(int j = 0; j < input.length; j++)
			{
				p.write(output[j].getName() + "\n");
			}
			
			p.write("\n");
		}
		
		p.close();
		
		if (TargetSentimentGlobal.DEBUG)
		{
			System.out.println("\n");
		}
		System.out.println(MODELNAME + " Evaluation Completed");
		
		
		/*
		try {
			PredictionAccuracy(filename_standard, filename_output);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/
		
	}
	
	
	
	public static void PredictionAccuracy(String standard, String output) throws FileNotFoundException
	{
		Scanner scan_standard = new Scanner(new File(standard));
		Scanner scan_output = new Scanner(new File(output));
		int counter = 0;
		int correct = 0;
		int sentence = 1;
		
		int standard_offset =8;
		int output_offset=0;
		
		int tp = 0;
		int tn = 0;
		int fp = 0;
		int fn = 0;
		
		while(scan_standard.hasNextLine() && scan_output.hasNextLine())
		{
			String line_standard = scan_standard.nextLine().trim().toLowerCase();
			String line_output = scan_output.nextLine().trim().toLowerCase();
			
			if (line_standard.equals(""))
			{
				sentence++;
				continue;
			}
			counter++;
			
			if (line_standard.equals(line_output))
				correct++;
			else
			{
				String[] lines_standard = line_standard.split("\t");
				String[] lines_output = line_output.split("\t");
				
				String standard_result = lines_standard[standard_offset].trim();
				String output_result = lines_output[output_offset].trim();
				
				System.out.println(standard_result + "\t" + output_result);
				if (standard_result.equals(output_result))
				{
					//System.out.println(line_standard + "\t" + line_output);
					correct++;
				}
			}
			
		}
		
		scan_standard.close();
		scan_output.close();
		
		System.out.println("In " + sentence + " sentences, Word Counter = " + counter + "\tWord Correct = " + correct +"\tAccuracy = " + (correct + 0.0) / counter);
	}

	
	public void Visualize(String[] args)
	{
		int network_index = 0;
		
		if (args.length > 0)
			network_index = Integer.parseInt(args[0]);
		
		TableLookupNetwork network = (TableLookupNetwork)this.networks[network_index];
		
		System.out.println("Networks size = " + this.networks.length);
		
		/*PipeTargetSentimentViewer viewer = new PipeTargetSentimentViewer(this.compiler, this.fm, 5);
		
		viewer.visualizeNetwork(network, null, "Sentiment Model");
		*/
		//viewer.saveImage("testimage");
		
		
	}
	
	
	public void loadWeightfromFile(String filename)
	{
		Scanner scanner = null;
		boolean started = false;
		String type, output, input, line;
		double weight;
		Integer featureId = -1;
		boolean found = true;
		
		try {
			scanner = new Scanner(new File(filename));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		HashMap<String, HashMap<String, HashMap<String, Integer>>> featureIntMap = param.getFeatureIntMap();
		param.unlock();
		
		//System.out.println(featureIntMap.keySet());
		System.out.println("Reading weights from paper's file");
		
		while(scanner.hasNextLine())
		{
			line = scanner.nextLine().trim();
			
			if (line.length() == 0)
				continue;
			
			if (line.startsWith("relations:"))
			{
				started = false;
			}
			
			if (started)
			{
				String[] temp = line.split("=");
				output = temp[1].trim();
				
				int t = temp[0].lastIndexOf("_");
				input = temp[0].substring(t);
				input = input.substring(0, input.length() - 1);
				
				type = temp[0].substring(0, t);
				weight = Double.parseDouble(temp[2]);
				
				
				found = false;
				HashMap<String, HashMap<String, Integer>> outputInputMap = featureIntMap.get(type);
				if (outputInputMap != null)
				{
					HashMap<String, Integer> inputMap = outputInputMap.get(output);
					if (inputMap != null)
					{
						featureId = inputMap.get(input);
						if (featureId != null)
						{
							found = true;
							
						}
					}
					
				}
				

				
				if (found)
				{
					//param.setWeight(featureId, weight);
				}
				else
				{
					featureId = param.toFeature(type, output, input);
					//param.setWeight(featureId, weight);
			
				}
				
				
			}
			
			
			if (line.startsWith("features:"))
			{
				started = true;
			}
			
		}
		
		scanner.close();
		
		param.lockItAndKeepExistingFeatureWeights();
		
		
		try {
			scanner = new Scanner(new File(filename));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		while(scanner.hasNextLine())
		{
			line = scanner.nextLine().trim();
			
			if (line.length() == 0)
				continue;
			
			if (line.startsWith("relations:"))
			{
				started = false;
			}
			
			if (started)
			{
				String[] temp = line.split("=");
				output = temp[1].trim();
				
				int t = temp[0].lastIndexOf("_");
				input = temp[0].substring(t);
				input = input.substring(0, input.length() - 1);
				
				type = temp[0].substring(0, t);
				weight = Double.parseDouble(temp[2]);
				
				
				found = false;
				HashMap<String, HashMap<String, Integer>> outputInputMap = featureIntMap.get(type);
				if (outputInputMap != null)
				{
					HashMap<String, Integer> inputMap = outputInputMap.get(output);
					if (inputMap != null)
					{
						featureId = inputMap.get(input);
						if (featureId != null)
						{
							found = true;
							
						}
					}
					
				}
				

				
				if (found)
				{
					param.setWeight(featureId, weight);
				}
				else
				{
					System.err.println("Features not found!!!!!!");
					//featureId = param.toFeature(type, output, input);
					//param.setWeight(featureId, weight);
			
				}
				
				
			}
			
			
			if (line.startsWith("features:"))
			{
				started = true;
			}
			
		}
		
		
		scanner.close();
		
		
		this.printFeature();
	}
	
	
	
	public void Train(String[] args)
	{	
		trainingMode = true;
		//defined by user
		initTraining(args);
		
		//defined by user		
		initInstanceParser(this);		

		
		try {
			//defined by user
			parser.BuildInstances();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		//defined by user
		initFeatureManager();
		//defined by user
		initNetworkCompiler();
		
		networkmodel = NetworkConfig.TRAIN_MODE_IS_GENERATIVE ? GenerativeNetworkModel.create(fm, compiler)
				: DiscriminativeNetworkModel.create(fm, compiler);
		
		try {
			this.resetTimer();
			long time = System.currentTimeMillis();
			if (!this.demoMode)
			{
				networkmodel.train(Instances, MaxIteration);
			}
			else // only use 1 process instead of multi-thread to read networks without training
			{
				inst_features = new int[Instances.length][][];
				
				for (int k = 0; k < Instances.length; k++) {
					Instances[k].setInstanceId(k + 1);
					InputToken[] inputs = (InputToken[])Instances[k].getInput();
					inst_features[k] = new int[inputs.length][];
				}
				
				
				

				// create the threads.
				LocalNetworkLearnerThread learners = new LocalNetworkLearnerThread(0,
						fm, Instances, compiler, -1);
				// distribute the works into different threads.
				// WARNING: must do the following sequentially..

				learners.touch();

				// finalize the features.
				fm.getParam_G().lockIt();

				networks = learners.getNetworks();

			}
			time = System.currentTimeMillis() - time;
			this.timer += time;
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		System.out.println("Total time: " + this.timer / 1000.0);
		
		try {
			//defined by user
			if (!demoMode)
			{
				saveModel();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		String filename_model = (String) getParameters("filename_model");
		
		PrintWriter modelTextWriter = null;
		try {
			modelTextWriter = new PrintWriter(filename_model + ".train");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		for (int k = 0; k < Instances.length; k++) {
			InputToken[] inputs = (InputToken[])Instances[k].getInput();
			OutputToken[] outputs = (OutputToken[])Instances[k].getOutput();
			for(int i = 0; i < inputs.length; i++)
			{
				String output_features = "";
				
				for(int j = 0; j < inst_features[k][i].length; j++)
				{
					output_features += " " + inst_features[k][i][j];
				}
				
				String output_labels = "";
				
				String output = outputs[i].getName();
				if (output.equals("O"))
				{
					output_labels = "O _ ";
				}
				else
				{
					output_labels = output.charAt(0) + " " + output.substring(2) + " ";
				}
				
				
				modelTextWriter.println(output_labels + "----" + output_features);
				
			}
			
			modelTextWriter.println();
		}
		
		modelTextWriter.close();
		
		printFeature();
	}
	
	
	public void Evaluate(String[] args)
	{
		trainingMode = false;
		
		Instance[] instances_outputs = null;

		//defined by user
		initEvaluation(args);
		
		if (!demoMode)
		{
			

			try {
				//defined by user
				loadModel();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			//defined by user
			initFeatureManager();
			//defined by user
			initNetworkCompiler();
		}
		
		//defined by user
		initInstanceParser(this);

		
		try {
			//defined by user
			parser.BuildInstances();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		networkmodel = NetworkConfig.TRAIN_MODE_IS_GENERATIVE ? GenerativeNetworkModel.create(fm, compiler)
				: DiscriminativeNetworkModel.create(fm, compiler);
		
		/*
		for(int k = 0; k < Instances.length; k++){
			//Instances[k].removeOutput();
			Instances[k].setUnlabeled();
		}*/
		
		inst_features = new int[Instances.length][][];
		
		for (int k = 0; k < Instances.length; k++) {
			Instances[k].setInstanceId(k + 1);
			InputToken[] inputs = (InputToken[])Instances[k].getInput();
			inst_features[k] = new int[inputs.length][];
		}
		
		
		try {
			this.resetTimer();
			long time = System.currentTimeMillis();
			instances_outputs = networkmodel.decode(Instances);
			time = System.currentTimeMillis() - time;
			this.timer += time;
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		System.out.println("Total time: " + this.timer / 1000.0);
		
		//defined by user
		//evaluationResult(instances_outputs);
		
		
		String filename_model = (String) getParameters("filename_model");
		
		PrintWriter modelTextWriter = null;
		try {
			modelTextWriter = new PrintWriter(filename_model + ".test");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		for (int k = 0; k < Instances.length; k++) {
			InputToken[] inputs = (InputToken[])Instances[k].getInput();
			OutputToken[] outputs = (OutputToken[])Instances[k].getOutput();
			for(int i = 0; i < inputs.length; i++)
			{
				String output_features = "";
				
				for(int j = 0; j < inst_features[k][i].length; j++)
				{
					output_features += " " + inst_features[k][i][j];
				}
				
				String output_labels = "";
				
				String output = outputs[i].getName();
				if (output.equals("O"))
				{
					output_labels = "O _ ";
				}
				else
				{
					output_labels = output.charAt(0) + " " + output.substring(2) + " ";
				}
				
				
				modelTextWriter.println(output_labels + "----" + output_features);
				
			}
			
			modelTextWriter.println();
		}
		
		modelTextWriter.close();
		
		
	}
	




	

}

package com.nlp.algomodel;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import com.nlp.algomodel.parser.InstanceParser;
import com.nlp.commons.ml.opt.Optimizer;
import com.nlp.commons.types.Instance;
import com.nlp.hybridnetworks.DiscriminativeNetworkModel;
import com.nlp.hybridnetworks.FeatureManager;
import com.nlp.hybridnetworks.GenerativeNetworkModel;
import com.nlp.hybridnetworks.GlobalNetworkParam;
import com.nlp.hybridnetworks.LocalNetworkLearnerThread;
import com.nlp.hybridnetworks.Network;
import com.nlp.hybridnetworks.NetworkCompiler;
import com.nlp.hybridnetworks.NetworkConfig;
import com.nlp.hybridnetworks.NetworkModel;



public abstract class AlgoModel {
	
	public static String MODELNAME = "Abstract";
	
		
	public Instance[] Instances;
	protected Network[] networks;
	protected int MaxIteration = 0;
	
	protected InstanceParser parser;
	protected NetworkCompiler compiler;
	protected FeatureManager fm;
	//protected Visualizer visualizer;
	protected NetworkModel networkmodel;
	public GlobalNetworkParam param;
	
	protected long timer = 0;
	
	protected HashMap<String, Object> parameters = new HashMap<String, Object>();
	
	protected boolean demoMode = false;
	protected boolean trainingMode = true;
	
	public AlgoModel()
	{
		param = new GlobalNetworkParam();
	}

	/***
	 * Initialize InstanceParser for a specific algorithm model
	 */
	protected abstract void initInstanceParser(AlgoModel algomodel);
	
	
	/***
	 * Initialize NetworkCompiler for a specific algorithm model
	 * It will call getNetworkCompilerParameters() from Parser to get necessary parameters
	 */
	protected abstract void initNetworkCompiler();
	
	/***
	 * Initialize FeatureManager for a specific algorithm model
	 * It will call getFeatureMgrParameters() from Parser to get necessary parameters
	 */
	protected abstract void initFeatureManager();
	
	/**
	 * Save the trained model into disk
	 * @throws IOException
	 */
	protected abstract void saveModel() throws IOException;
	
	/**
	 * Load the trained model into memory
	 * @throws IOException
	 */
	protected abstract void loadModel() throws IOException;
	
	
	/**
	 * Grather precision recall info
	 * @param instances_outputs
	 */
	protected abstract void evaluationResult(Instance[] instances_outputs);
	
	/**
	 * Initialize training stuff, including argument parsing, variable initialization
	 */
	protected abstract void initTraining(String[] args);
	
	
	/**
	 *  Initialize evaluation stuff, including argument parsing, variable initialization
	 */
	protected abstract void initEvaluation(String[] args);

	
	
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
				for (int k = 0; k < Instances.length; k++) {
					Instances[k].setInstanceId(k + 1);
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
		
		for(int k = 0; k < Instances.length; k++){
			//Instances[k].removeOutput();
			Instances[k].setUnlabeled();
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
		evaluationResult(instances_outputs);
		
		
	}
	
	public void Visualize(String[] args)
	{
		
	}
	
	public void Execute(String[] args) 
	{
		String ExecutionMode = args[0];
		String[] newargs = reduceArgs(args);
		
		if (ExecutionMode.equals("TRAIN"))
		{
			this.Train(newargs);
		}
		else if (ExecutionMode.equals("EVALUATE"))
		{
			this.Evaluate(newargs);			
		}
		else if (ExecutionMode.equals("VISUALIZE"))
		{
			
		}
		else
		{
			
		}
	}
	
	public void LoadModel(String[] args)
	{
		initEvaluation(args);

		try {
			loadModel();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	
		initFeatureManager();
		initNetworkCompiler();
	}
	
	/**
	 * Get rid of 1st Argument and return the rest
	 * @param args
	 * @return
	 */
	static String[] reduceArgs(String[] args)
	{
		String[] restargs = new String[args.length - 1];
		
		for(int i = 1; i < args.length; i++)
			restargs[i - 1] = args[i];
		
		return restargs;
		
	}
	
	
	public FeatureManager getFeatureManager()
	{
		return this.fm;
	}
	
	public void resetTimer()
	{
		this.timer = 0;
	}
	
	public long getTimer()
	{
		return this.timer;
	}
	
	public void setParameters(String key, Object value)
	{
		this.parameters.put(key, value);
	}
	
	public Object getParameters(String key)
	{
		return this.parameters.get(key);
	}
	
	public void setDemoMode(boolean b)
	{
		this.demoMode = b;
	}
	
	//return true if we are going to use the instance
	public boolean useTheIntance(Instance instance)
	{
		return true;
	}

	public void setOptimizer(Optimizer opt) {
		// TODO Auto-generated method stub
		this.param.setOptimzer(opt);
	}
	
	
}

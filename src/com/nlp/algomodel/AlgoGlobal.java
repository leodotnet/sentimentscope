package com.nlp.algomodel;

public class AlgoGlobal {

	public AlgoGlobal() {
		// TODO Auto-generated constructor stub
	}
	
	public static enum TaskMode {TRAIN, EVALUATE};
	
	public static String ALGO_MODEL_PATH = "com.nlp.algomodel.AlgoModel";
	
	public static Class parseClass(String modelname)
	{
		Class model = null;
		if (modelname.equals("sentimentspan_latent"))
		{
			model = com.nlp.targetedsentiment.f.latent.TargetSentimentAlgoModel.class;
		}
		
		else if (modelname.equals("sentimentspan_nonlatent"))
		{
			model = com.nlp.targetedsentiment.f.TargetSentimentAlgoModel.class;
		}
		
		else if (modelname.equals("semimarkov_nonlatent"))
		{
			model = com.nlp.targetedsentiment.f.semimarkov.TargetSentimentAlgoModel.class;
		}
		else if (modelname.equals("semimarkov_latent"))
		{
			model = com.nlp.targetedsentiment.f.semimarkov.latent.TargetSentimentAlgoModel.class;
		}
		else if (modelname.equals("baseline_collapse"))
		{
			model = com.nlp.targetedsentiment.f.baseline.PipeTargetSentimentAlgoModel.class;
		}
		else if (modelname.equals("baseline_pipelineNE"))
		{
			model = com.nlp.targetedsentiment.f.pipelineNE.PipeTargetSentimentAlgoModel.class;
		}
		else if (modelname.equals("baseline_pipelineSent"))
		{
			model = com.nlp.targetedsentiment.f.pipelineSent.PipeTargetSentimentAlgoModel.class;
		}
		else if (modelname.equals("baseline_joint"))
		{
			model = com.nlp.targetedsentiment.f.baseline.joint.PipeTargetSentimentAlgoModel.class;
		}
		else if (modelname.equals("baseline_joint_exact"))
		{
			model = com.nlp.targetedsentiment.f.baseline.exactjoint.PipeTargetSentimentAlgoModel.class;
		}
		
		
		return model;
	}

}

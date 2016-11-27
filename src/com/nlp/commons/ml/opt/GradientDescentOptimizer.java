package com.nlp.commons.ml.opt;

import java.util.ArrayList;
import java.util.Arrays;

import com.nlp.commons.ml.opt.LBFGS.ExceptionWithIflag;
import com.nlp.targetedsentiment.util.TargetSentimentGlobal;

public class GradientDescentOptimizer extends Optimizer {

	double learningRate = 1e-4;
	static boolean DEBUG = false;
	static int good_counter = 0;
	public static boolean fixLearningRate = true;
	
	ArrayList<Double> latest_obj = new ArrayList<Double>();
	public static int latest_obj_size = 4;
	
	public GradientDescentOptimizer(double learningRate) {
		this.learningRate = learningRate;
	}

	@Override
	public boolean optimize() throws ExceptionWithIflag {
		if (DEBUG)
		{
			System.out.println("grad_b: "+ Arrays.toString(_g));
			System.out.println("weight_b: " + Arrays.toString(_x));
		}
		
		for(int i = 0; i < this._x.length; i++)
		{
			this._x[i] -= this._g[i] * learningRate;
			
		}
		
		if (DEBUG)
		{
			System.out.println("weight_a: " + Arrays.toString(_x));
		}
		return false;
	}
	
	public void setLearningRate(double learningRate)
	{
		this.learningRate = learningRate;
	}
	
	public void halfLearningRate()
	{
		this.learningRate *= 0.95;
	}
	
	public void increaseLearningRate()
	{
		this.learningRate *= 1.1;
	}
	
	public double getLearningRate()
	{
		return this.learningRate;
	}
	
	public void updateLearningRate(double obj)	
	{
		if (fixLearningRate)
			return;
		
		if (latest_obj.size() >= latest_obj_size)
		{
			latest_obj.remove(0);
		}
		
		latest_obj.add(obj);
		
		boolean bad = true;
		boolean good = true;
		for(int i = 0; i < latest_obj.size() - 1; i++)
		{
			if (obj > latest_obj.get(i))
			{
				bad = false;
				break;
			}
			
			if (obj < latest_obj.get(i))
			{
				good = false;
			}
		}
		
		
		if (bad)
		{
			halfLearningRate();
			//System.out.println("new LearningRate-:" + TargetSentimentGlobal.gd_opt.getLearningRate());
		}
		
		/*
		if (good)
		{
			good_counter++;
		}
		else
		{
			good_counter = 0;
		}
		
		if (good_counter > 10)
		{
			this.increaseLearningRate();
			good_counter = 0;
			System.out.println("new LearningRate+:" + TargetSentimentGlobal.gd_opt.getLearningRate());
		}*/
		
	}

}

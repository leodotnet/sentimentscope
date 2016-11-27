package com.nlp.commons.ml.gm.linear;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import com.nlp.commons.types.InputToken;
import com.nlp.commons.types.Instance;

//WARNING: to use this, please make sure there is strictly no overlapping, and no empty spans.
public class LinearSpanInstance extends Instance{
	
	private static final long serialVersionUID = 7130292676797965956L;
	
	protected InputToken[] _input;
	protected LinearSpan[] _output;
	protected LinearSpan[] _prediction;
	
	public LinearSpanInstance(int instanceId, double weight, InputToken[] input) {
		super(instanceId, weight);
		this._input = input;
	}
	
	public LinearSpanInstance(int instanceId, double weight, InputToken[] input, LinearSpan[] output) {
		super(instanceId, weight);
		this._input = input;
		this._output = output;
	}
	
	@Override
	public int size(){
		return this._input.length;
	}
	
	@Override
	public Instance duplicate() {
		return new LinearSpanInstance(this._instanceId, this._weight, this._input, this._output);
	}
	
	@Override
	public void removeOutput() {
		this._output = null;
	}

	@Override
	public void removePrediction() {
		this._prediction = null;
	}

	@Override
	public InputToken[] getInput() {
		return this._input;
	}

	@Override
	public LinearSpan[] getOutput() {
		return this._output;
	}

	@Override
	public LinearSpan[] getPrediction() {
		return this._prediction;
	}

	@Override
	public boolean hasOutput() {
		return this._output != null;
	}
	
	@Override
	public boolean hasPrediction() {
		return this._prediction != null;
	}
	
	@Override
	public void setPrediction(Object o) {
		this._prediction = (LinearSpan[]) o;
	}
	
	public int countNumOutputs(){
		int count = 0;
		for(LinearSpan output : this._output){
			if(output.getOutputToken().getName().endsWith("-L")||
					output.getOutputToken().getName().endsWith("-I")){
			} else {
				count ++;
			}
		}
		return count;
	}
	
	public int countNumPredictions(){
		int count = 0;
		for(LinearSpan output : this._prediction){
			if(output.getOutputToken().getName().endsWith("-L")||
					output.getOutputToken().getName().endsWith("-I")){
			} else {
				count ++;
			}
		}
		return count;
	}
	
	public int countNumCorrectlyPredicted(){
		Arrays.sort(this._output);
		Arrays.sort(this._prediction);
		
		ArrayList<LinearSpan> spans_output = new ArrayList<LinearSpan>();
		for(int k = 0; k<this._output.length; k++){
			LinearSpan output = this._output[k];
			if(output.getOutputToken().getName().endsWith("-B")){
				int bIndex = output.getBIndex();
				boolean found_last = false;
				for(int i = k+1; i<this._output.length; i++){
					LinearSpan output2 = this._output[i];
					if(output2.getOutputToken().getName().endsWith("-L")){
						found_last = true;
						int eIndex = output2.getEIndex();
						LinearSpan span = new LinearSpan(bIndex, eIndex, output.getOutputToken());
						spans_output.add(span);
						k=i;
						break;
					}
				}
				if(!found_last){
					throw new RuntimeException("There should be the last tag, but did not find it!");
				}
			} else {
				spans_output.add(output);
			}
		}
		
		ArrayList<LinearSpan> spans_prediction = new ArrayList<LinearSpan>();
		for(int k = 0; k<this._prediction.length; k++){
			LinearSpan prediction = this._prediction[k];
			if(prediction.getOutputToken().getName().endsWith("-B")){
				int bIndex = prediction.getBIndex();
				boolean found_last = false;
				for(int i = k+1; i<this._prediction.length; i++){
					LinearSpan prediction2 = this._prediction[i];
					if(prediction2.getOutputToken().getName().endsWith("-L")){
						found_last = true;
						int eIndex = prediction2.getEIndex();
						LinearSpan span = new LinearSpan(bIndex, eIndex, prediction.getOutputToken());
						spans_prediction.add(span);
						k=i;
						break;
					}
				}
				if(!found_last){
					System.err.println("There should be the last tag, but did not find it!");
				}
			} else if(prediction.getOutputToken().getName().endsWith("-I")){
				//ignore...
			} else {
				spans_prediction.add(prediction);
			}
		}
		
		int count = 0;
		for(LinearSpan span : spans_output){
			int key = Collections.binarySearch(spans_prediction, span);
			if(key >= 0)
				count++;
		}
		return count;
	}
	
}

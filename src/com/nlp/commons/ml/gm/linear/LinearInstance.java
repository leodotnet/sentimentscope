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

import com.nlp.commons.types.InputToken;
import com.nlp.commons.types.Instance;
import com.nlp.commons.types.OutputToken;

public class LinearInstance extends Instance{
	
	private static final long serialVersionUID = 3336336220436168888L;
	
	protected InputToken[] _input;
	protected OutputToken[] _output;
	protected OutputToken[] _prediction;
	
	public LinearInstance(int instanceId, double weight, InputToken[] inputs){
		super(instanceId, weight);
		this._input = inputs;
	}
	
	public LinearInstance(int instanceId, double weight, InputToken[] input, OutputToken[] output){
		super(instanceId, weight);
		this._input = input;
		this._output = output;
	}
	
	public LinearInstance(int instanceId, double weight, InputToken[] input, OutputToken[] output, OutputToken[] prediction){
		super(instanceId, weight);
		this._input = input;
		this._output = output;
		this._prediction = prediction;
	}
	
	@Override
	public InputToken[] getInput(){
		return this._input;
	}
	
	@Override
	public OutputToken[] getOutput(){
		return this._output;
	}
	
	@Override
	public OutputToken[] getPrediction() {
		return this._prediction;
	}

	public int size(){
		return this._input.length;
	}
	
	@Override
	public void removeOutput(){
		this._output = null;
	}
	
	@Override
	public void removePrediction() {
		this._prediction = null;
	}
	
	@Override
	public LinearInstance duplicate() {
		LinearInstance instance= new LinearInstance(this._instanceId, this._weight, this._input, this._output);
		instance._prediction = this._prediction;
		return instance;
	}
	
	@Override
	public void setPrediction(Object o){
		OutputToken[] prediction = (OutputToken[]) o;
		if(this._output.length!= prediction.length)
			throw new RuntimeException("The length of the correct and predicted tags do not match."+this._output.length+"/"+prediction.length);
		this._prediction = prediction;
	}
	
	@Override
	public boolean hasOutput() {
		return this._output!=null;
	}
	
	@Override
	public boolean hasPrediction() {
		return this._prediction!=null;
	}
	
	public int countNumCorrectlyPredicted(){
		if(this._output == null)
			throw new RuntimeException("This instance has no outputs.");
		if(this._prediction == null)
			throw new RuntimeException("This instance has no predictions.");
		
		int count = 0;
		for(int k = 0; k<this._output.length; k++){
			OutputToken corr = this._output[k];
			OutputToken pred = this._prediction[k];
			if(corr.equals(pred)){
				count ++;
			}
		}
		return count;
	}

}
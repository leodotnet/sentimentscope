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
package com.nlp.commons.types;

import java.io.Serializable;

public abstract class Instance implements Serializable{
	
	private static final long serialVersionUID = 4998596827132890817L;
	
	protected int _instanceId;
	protected double _weight;
	protected boolean _isLabeled;
	
	//the instance id should not be zero.
	public Instance(int instanceId, double weight){
		if(instanceId==0)
			throw new RuntimeException("The instance id is "+instanceId);
		this._instanceId = instanceId;
		this._weight = weight;
	}
	
	public void setInstanceId(int instanceId){
		this._instanceId = instanceId;
	}
	
	public int getInstanceId(){
		return this._instanceId;
	}
	
	public double getWeight(){
		return this._weight;
	}
	
	public void setWeight(double weight){
		this._weight = weight;
	}
	
	public abstract int size();
	
	public boolean isLabeled(){
		return this._isLabeled;
	}
	
	public void setLabeled(){
//		if(this.getOutput()==null){
//			throw new RuntimeException("This instance has no outputs, but you want to make it labeled??");
//		}
		this._isLabeled = true;
	}
	
	public void setUnlabeled(){
		this._isLabeled = false;
	}
	
	public abstract Instance duplicate();
	
	public abstract void removeOutput();
	public abstract void removePrediction();
	
	public abstract Object getInput();
	public abstract Object getOutput();
	public abstract Object getPrediction();
	
	public abstract boolean hasOutput();
	public abstract boolean hasPrediction();
	
	public abstract void setPrediction(Object o);
	
}
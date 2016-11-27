package com.nlp.commons.ml.gm.linear;

import java.io.Serializable;

import com.nlp.commons.types.OutputToken;

public class LinearSpan implements Serializable, Comparable<LinearSpan>{
	
	private static final long serialVersionUID = -4877978104403954289L;
	
	private int _bIndex;
	private int _eIndex;
	private OutputToken _output;
	
	public LinearSpan(int bIndex, int eIndex, OutputToken output){
		this._bIndex = bIndex;
		this._eIndex = eIndex;
		this._output = output;
	}
	
	public int length(){
		return this._eIndex - this._bIndex;
	}
	
	public int getBIndex(){
		return this._bIndex;
	}
	
	public int getEIndex(){
		return this._eIndex;
	}
	
	public OutputToken getOutputToken(){
		return this._output;
	}
	
	@Override
	public boolean equals(Object o){
		if(o instanceof LinearSpan){
			LinearSpan span = (LinearSpan)o;
			return this._bIndex == span._bIndex
					&& this._eIndex == span._eIndex
					&& this._output.equals(span._output);
		}
		return false;
	}
	
	@Override
	public String toString(){
		return "LinearSpan:["+this._bIndex+","+this._eIndex+"]"+this._output;
	}

	@Override
	public int compareTo(LinearSpan span) {
		if(this._bIndex != span._bIndex)
			return this._bIndex - span._bIndex;
		if(this._eIndex != span._eIndex)
			return this._eIndex - span._eIndex;
		return this._output.getId() - span._output.getId();
	}
	
}

package com.nlp.commons.ml.gm.linear;

import java.io.Serializable;
import java.util.Arrays;

import com.nlp.commons.types.OutputToken;

public class OutputTokenArray implements Serializable{
	
	private static final long serialVersionUID = 8181520592866135065L;
	
	protected OutputToken[] _tokens;
	
	public OutputTokenArray(OutputToken[] tokens){
		this._tokens = tokens;
	}
	
	public OutputToken[] getTokens(){
		return this._tokens;
	}
	
	public int size(){
		return this._tokens.length;
	}
	
	@Override
	public boolean equals(Object o){
		if(o instanceof OutputTokenArray){
			OutputTokenArray a = (OutputTokenArray)o;
			return Arrays.equals(this._tokens, a._tokens);
		}
		return false;
	}
	
	@Override
	public int hashCode(){
		return Arrays.hashCode(this._tokens);
	}

	public OutputTokenArray getSuffix(){
		OutputToken[] suffix = new OutputToken[this._tokens.length-1];
		for(int k = 0; k<suffix.length; k++)
			suffix[k] = this._tokens[k+1];
		return new OutputTokenArray(suffix);
	}

	public OutputTokenArray getPrefix(){
		OutputToken[] prefix = new OutputToken[this._tokens.length-1];
		for(int k = 0; k<prefix.length; k++)
			prefix[k] = this._tokens[k];
		return new OutputTokenArray(prefix);
	}
	
	@Override
	public String toString(){
		return "OutputTokenArray:"+Arrays.toString(this._tokens);
	}
	
}
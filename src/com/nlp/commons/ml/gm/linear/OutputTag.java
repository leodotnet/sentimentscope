package com.nlp.commons.ml.gm.linear;

import com.nlp.commons.types.OutputToken;

//a simple output tag.

public class OutputTag extends OutputToken{
	
	private static final long serialVersionUID = -522854594328831053L;
	
	public OutputTag(String name) {
		super(name);
	}
	
	@Override
	public boolean equals(Object o) {
		if(o instanceof OutputTag){
			OutputTag tag = (OutputTag)o;
			return this._name.equals(tag._name);
		}
		return false;
	}

	@Override
	public int hashCode() {
		return this._name.hashCode() + 7;
	}
	
	public boolean isBegin(){
		return this._name.endsWith("-B");
	}
	
	public boolean isInside(){
		return this._name.endsWith("-I");
	}
	
	public boolean isLast(){
		return this._name.endsWith("-L");
	}
	
	public boolean isOutside(){
		return this._name.equals("O");
	}
	
	public boolean isUnit(){
		return this._name.equals("-U");
	}
	
	@Override
	public String toString(){
		return "OutputTag:"+this._name;
	}
	
}

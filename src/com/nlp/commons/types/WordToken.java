package com.nlp.commons.types;


public class WordToken extends InputToken{
	
	private static final long serialVersionUID = -1296542134339296118L;
	
	public WordToken(String name) {
		super(name);
	}
	
	@Override
	public boolean equals(Object o) {
		if(o instanceof WordToken){
			WordToken w = (WordToken)o;
			return w._name.equals(this._name);
		}
		return false;
	}
	
	@Override
	public int hashCode() {
		return this._name.hashCode() + 7;
	}
	
	@Override
	public String toString() {
		return "WORD:"+this._name;
	}
	
}
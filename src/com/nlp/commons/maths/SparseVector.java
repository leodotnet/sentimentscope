package com.nlp.commons.maths;

import java.io.Serializable;

public class SparseVector implements Serializable{
	
	private static final long serialVersionUID = 281435669943512474L;
	
	private DenseVector _dv;
	private int[] _ids;
	
	private SparseVector(DenseVector dv, int[] ids){
		this._dv = dv;
		this._ids = ids;
	}
	
	public int[] getIds(){
		return this._ids;
	}
	
	public int length(){
		return this._ids.length;
	}
	
	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		for(int k = 0; k< this._ids.length; k++){
			if(k!=0)
				sb.append(' ');
			sb.append(this._ids[k]);
			sb.append(':');
			sb.append(this._dv);
		}
		return sb.toString();
	}
	
}

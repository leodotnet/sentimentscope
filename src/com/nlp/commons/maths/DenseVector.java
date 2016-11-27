package com.nlp.commons.maths;

import java.io.Serializable;

public class DenseVector implements Serializable{
	
	private static final long serialVersionUID = 2371799875808032043L;
	private static final int defaultCapacity = 1000;
	private static final double growRate = 2.0; // must be > 1
	
	private double[] _vals;
	private int _capacity;
	private int _size;
	
	public DenseVector(){
		this(defaultCapacity);
	}
	
	public DenseVector(int capacity){
		this._size = 0;
		this._capacity = capacity;
		this._vals = new double[capacity];
	}
	
	//add a value to the vector, and return its index.
	public int add(double v){
		if(this._size>=this._capacity)
			this.grow();
		this._vals[this._size++] = v;
		return this._size - 1;
	}
	
	public void grow(){
		int newCapacity = (int)(this._capacity*growRate);
		double[] vals = new double[newCapacity];
		System.arraycopy(this._vals, 0, vals, 0, this._capacity);
		this._vals = vals;
		this._capacity = newCapacity;
	}
	
	public void shrink(){
		double[] vals = new double[this._size];
		System.arraycopy(this._vals, 0, vals, 0, this._size);
		this._vals = vals;
		this._capacity = this._size;
	}
	
	//get the value based on the id.
	public double get(int id){
		return this._vals[id];
	}
	
	public int capacity(){
		return this._capacity;
	}
	
	public int size(){
		return this._size;
	}
	
	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append("Capacity:"+this._capacity);
		sb.append('\n');
		sb.append("Size:"+this._size);
		sb.append('\n');
		sb.append('[');
		for(int k = 0; k<this._vals.length; k++){
			if(k!=0)
				sb.append(',');
			sb.append(k);
			sb.append(':');
			sb.append(this._vals[k]);
		}
		sb.append(']');
		return sb.toString();
	}
	
}

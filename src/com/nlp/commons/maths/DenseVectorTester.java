package com.nlp.commons.maths;

public class DenseVectorTester {
	
	public static void main(String args[]){
		
		DenseVector dv = new DenseVector();
		
		for(int k = 0 ; k<1000; k++){
			double v = Math.random();
			dv.add(v);
		}
		
	}

}

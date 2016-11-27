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
package com.nlp.hybridnetworks;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;

import com.nlp.commons.ml.opt.LBFGSOptimizer;
import com.nlp.commons.ml.opt.MathsVector;
import com.nlp.commons.ml.opt.LBFGS.ExceptionWithIflag;
import com.nlp.commons.ml.opt.Optimizer;
import com.nlp.targetedsentiment.util.WordEmbedding;

//TODO: other optimization and regularization methods. Such as the L1 regularization.

public class GlobalNetworkParam implements Serializable{
	
	private static final long serialVersionUID = -1216927656396018976L;
	
	//these parameters are used for discriminative training using LBFGS.
	protected transient double _kappa;
	protected transient Optimizer _opt;
	protected transient double[] _counts;
	protected transient double _obj_old;
	protected transient double _obj;
	protected transient int _version;
	
	//feature type - output - input
	protected HashMap<String, HashMap<String, HashMap<String, Integer>>> _featureIntMap;
	//feature type - input
	protected HashMap<String, ArrayList<String>> _type2inputMap;
	
	protected String[][] _feature2rep;//three-dimensional array representation of the feature.
	protected double[] _weights;
	protected boolean _isDiscriminative;
	protected int _size;
	protected int _fixedFeaturesSize;
	protected boolean _locked = false;
	
	public GlobalNetworkParam(){
		this._locked = false;
		this._version = -1;
		this._size = 0;
		this._fixedFeaturesSize = 0;
		this._obj_old = Double.NEGATIVE_INFINITY;
		this._obj = Double.NEGATIVE_INFINITY;
		this._isDiscriminative = !NetworkConfig.TRAIN_MODE_IS_GENERATIVE;
		if(this.isDiscriminative()){
			this._opt = new LBFGSOptimizer();
			this._kappa = NetworkConfig.L2_REGULARIZATION_CONSTANT;
		}
		this._featureIntMap = new HashMap<String, HashMap<String, HashMap<String, Integer>>>();
		this._type2inputMap = new HashMap<String, ArrayList<String>>();
	}
	
	public HashMap<String, HashMap<String, HashMap<String, Integer>>> getFeatureIntMap(){
		return this._featureIntMap;
	}
	
	public double[] getWeights(){
		return this._weights;
	}
	
	public int countFeatures(){
		return this._size;
	}
	
	public int countFixedFeatures(){
		return this._fixedFeaturesSize;
	}
	
	public boolean isFixed(int f_global){
		return f_global < this._fixedFeaturesSize;
	}
	
	public String[] getFeatureRep(int f_global){
		return this._feature2rep[f_global];
	}
	
	public synchronized void addCount(int feature, double count){
		
		if(Double.isNaN(count)){
			throw new RuntimeException("count is NaN.");
		}
		
		if(this.isFixed(feature))
			return;
		//if the model is discriminative model, we will flip the sign for
		//the counts because we will need to use LBFGS.
		if(this.isDiscriminative()){
			this._counts[feature] -= count;
		} else {
			this._counts[feature] += count;
		}
		
	}
	
	public synchronized void addObj(double obj){
		this._obj += obj;
	}
	
	public double getObj(){
		return this._obj;
	}
	
	public double getObj_old(){
		return this._obj_old;
	}
	
	private double getCount(int f){
		return this._counts[f];
	}
	
	public double[] getCounts()
	{
		return this._counts;
	}
	
	public double getWeight(int f){
		//if the feature is just newly created, for example, return the initial weight, which is zero.
//		if(f>=this._weights.length)
//			return NetworkConfig.FEATURE_INIT_WEIGHT;
		return this._weights[f];
	}
	
	public synchronized void setWeight(int f, double weight){
		if(this.isFixed(f)) return;
		this._weights[f] = weight;
	}
	
	public synchronized void overRideWeight(int f, double weight){
		this._weights[f] = weight;
	}
	
	public void unlock(){
		if(!this.isLocked())
			throw new RuntimeException("This param is not locked.");
		this._locked = false;
	}
	
	public void unlockForNewFeaturesAndFixCurrentFeatures(){
		if(!this.isLocked())
			throw new RuntimeException("This param is not locked.");
		this.fixCurrentFeatures();
		this._locked = false;
	}
	
	public void fixCurrentFeatures(){
		this._fixedFeaturesSize = this._size;
	}
	
	private void expandFeaturesForGenerativeModelDuringTesting(){
		
//		this.unlockForNewFeaturesAndFixCurrentFeatures();
		
		//if it is a discriminative model, then do not expand the features.
		if(this.isDiscriminative()){
			return;
		}
		
		System.err.println("==EXPANDING THE FEATURES===");
		System.err.println("Before expansion:"+this.size());
		Iterator<String> types = this._featureIntMap.keySet().iterator();
		while(types.hasNext()){
			String type = types.next();
			HashMap<String, HashMap<String, Integer>> output2input = this._featureIntMap.get(type);
			ArrayList<String> inputs = this._type2inputMap.get(type);
			System.err.println("Feature of type "+type+" has "+inputs.size()+" possible inputs.");
			Iterator<String> outputs = output2input.keySet().iterator();
			while(outputs.hasNext()){
				String output = outputs.next();
				for(String input : inputs){
					this.toFeature(type, output, input);
				}
			}
		}
		System.err.println("After expansion:"+this.size());
		
//		this.lockIt();
	}
	

	public void lockItAndKeepExistingFeatureWeights(){
		
		Random r = new Random(NetworkConfig.RANDOM_INIT_FEATURE_SEED);
		
		if(this.isLocked()) return;
		
		if(NetworkConfig.TRAIN_MODE_IS_GENERATIVE){
			this.expandFeaturesForGenerativeModelDuringTesting();
		}
		
		double[] weights_new = new double[this._size];
		this._counts = new double[this._size];
		for(int k = 0; k<this._weights.length; k++){
			weights_new[k] = this._weights[k];
		}
		for(int k = this._weights.length ; k<this._size; k++){
			weights_new[k] = NetworkConfig.RANDOM_INIT_WEIGHT ? (r.nextDouble()-.5)/10 :
				NetworkConfig.FEATURE_INIT_WEIGHT;
		}
		this._weights = weights_new;
		this.resetCountsAndObj();
		
		this._feature2rep = new String[this._size][];
		Iterator<String> types = this._featureIntMap.keySet().iterator();
		while(types.hasNext()){
			String type = types.next();
			HashMap<String, HashMap<String, Integer>> output2input = this._featureIntMap.get(type);
			Iterator<String> outputs = output2input.keySet().iterator();
			while(outputs.hasNext()){
				String output = outputs.next();
				HashMap<String, Integer> input2id = output2input.get(output);
				Iterator<String> inputs = input2id.keySet().iterator();
				while(inputs.hasNext()){
					String input = inputs.next();
					int id = input2id.get(input);
					this._feature2rep[id] = new String[]{type, output, input};
				}
			}
		}
		this._version = 0;
		this._opt = new LBFGSOptimizer();
		this._locked = true;
		
		System.err.println(this._size+" features.");
		
	}
	
	//if it is locked it means no new features will be allowed.
	public void lockIt(){
		
		Random r = new Random(NetworkConfig.RANDOM_INIT_FEATURE_SEED);
		
		if(this.isLocked()) return;
		
		this.expandFeaturesForGenerativeModelDuringTesting();
		
		double[] weights_new = new double[this._size];
		this._counts = new double[this._size];
		for(int k = 0; k<this._fixedFeaturesSize; k++){
			weights_new[k] = this._weights[k];
		}
		for(int k = this._fixedFeaturesSize ; k<this._size; k++){
			weights_new[k] = NetworkConfig.RANDOM_INIT_WEIGHT ? (r.nextDouble()-.5)/10 :
				NetworkConfig.FEATURE_INIT_WEIGHT;
		}
		
		this._weights = weights_new;
		this.resetCountsAndObj();
		
		this._feature2rep = new String[this._size][];
		Iterator<String> types = this._featureIntMap.keySet().iterator();
		while(types.hasNext()){
			String type = types.next();
			HashMap<String, HashMap<String, Integer>> output2input = this._featureIntMap.get(type);
			Iterator<String> outputs = output2input.keySet().iterator();
			while(outputs.hasNext()){
				String output = outputs.next();
				HashMap<String, Integer> input2id = output2input.get(output);
				Iterator<String> inputs = input2id.keySet().iterator();
				while(inputs.hasNext()){
					String input = inputs.next();
					int id = input2id.get(input);
					this._feature2rep[id] = new String[]{type, output, input};
				}
			}
		}
		this._version = 0;
		this._opt = new LBFGSOptimizer();
		this._locked = true;
		
		System.err.println(this._size+" features.");
		
	}
	
	public int size(){
		return this._size;
	}
	
	public boolean isLocked(){
		return this._locked;
	}
	
	public int getVersion(){
		return this._version;
	}
	
	
	public int toFeature(String type, String output, String input){

//		if(type.equals("emission")){
//			System.err.println("XXX"+type+"\t"+output+"\t"+input);
//		}
		
//		if(input.indexOf("low_point")!=-1){
//			System.err.println(type+"\t"+output+"\t"+input);
//			System.exit(1);
//		}
		
		//if it is locked, then we might return a dummy feature
		//if the feature does not appear to be present.
		if(this.isLocked()){
			if(!this._featureIntMap.containsKey(type)){
				return -1;
			} else {
				HashMap<String, HashMap<String, Integer>> output2input = this._featureIntMap.get(type);
				if(!output2input.containsKey(output)){
					return -1;
				} else {
					HashMap<String, Integer> input2id = output2input.get(output);
					if(!input2id.containsKey(input)){
						return -1;
					} else {
						return input2id.get(input);
					}
				}
			}
		}
//		if(type.equals("emission")){
//			System.err.println("|||"+type+"\t"+output+"\t"+input);
//		}
		
		if(!this._featureIntMap.containsKey(type)){
			this._featureIntMap.put(type, new HashMap<String, HashMap<String, Integer>>());
		}
		
		HashMap<String, HashMap<String, Integer>> mapByType = this._featureIntMap.get(type);
		
		if(!mapByType.containsKey(output))
			mapByType.put(output, new HashMap<String, Integer>());
		
		HashMap<String, Integer> subMap = mapByType.get(output);
		if(!subMap.containsKey(input)){
			subMap.put(input, this._size++);
			if(!this._type2inputMap.containsKey(type)){
				this._type2inputMap.put(type, new ArrayList<String>());
			}
			ArrayList<String> inputs = this._type2inputMap.get(type);
			int index = Collections.binarySearch(inputs, input);
			if(index<0){
				inputs.add(-1-index, input);
			}
//			System.err.println(type+"\t"+inputs.size());
		}
//		if(type.equals("emission")){
//			System.err.println("<<<"+type+"\t"+output+"\t"+input);
//		}
			
		return subMap.get(input);
	}
	
	//globally update the parameters.
	public synchronized boolean update(){
		
		boolean r;
		if(this.isDiscriminative()){
			r = this.updateDiscriminative();
		} else {
			r = this.updateGenerative();
		}
		
		this._obj_old = this._obj;
		
		return r;
	}
	
	private boolean updateGenerative(){
		
//		HashMap<String, Double> word2count = new HashMap<String, Double>();
		
		Iterator<String> types = this._featureIntMap.keySet().iterator();
		while(types.hasNext()){
			String type = types.next();
			HashMap<String, HashMap<String, Integer>> output2input = this._featureIntMap.get(type);
			
			Iterator<String> outputs = output2input.keySet().iterator();
			while(outputs.hasNext()){
				String output = outputs.next();
				
				HashMap<String, Integer> input2feature;
				Iterator<String> inputs;
				
				double sum = 0;
				input2feature = output2input.get(output);
				inputs = input2feature.keySet().iterator();
				while(inputs.hasNext()){
					String input = inputs.next();
					int feature = input2feature.get(input);
					sum += this.getCount(feature);
					
//					if(output.indexOf("*n:PlaceName -> ({ ' death valley ' })")!=-1){
//						System.err.println(Arrays.toString(this.getFeatureRep(feature))+"\t"+Math.exp(this.getWeight(feature)));
//					}
					
//					if(type.equals("emission")){
//						if(!word2count.containsKey(input)){
//							word2count.put(input, 0.0);
//						}
//						double oldCount = word2count.get(input);
//						word2count.put(input, oldCount+this.getCount(feature));
//					}
				}
				
//				if(Math.abs(1-sum)>1E-12){
//					System.err.println("sum="+sum+"\t"+type+"\t"+output);
//				}
				
				input2feature = output2input.get(output);
				inputs = input2feature.keySet().iterator();
				while(inputs.hasNext()){
					String input = inputs.next();
					int feature = input2feature.get(input);
					double value = sum != 0 ? this.getCount(feature)/sum : 1.0/input2feature.size();
					this.setWeight(feature, Math.log(value));
					
//					if(value>1E-15)
//					{
//						String s = Arrays.toString(this.getFeatureRep(feature));
//						if(s.indexOf("transition")!=-1 && s.indexOf("low_point_1")!=-1){
//							System.err.println(s+"\t"+value);
//						}
//					}
					
					if(Double.isNaN(Math.log(value))){
						throw new RuntimeException("x"+value+"\t"+this.getCount(feature)+"/"+sum+"\t"+input2feature.size());
					}
				}
			}
		}
		boolean done = Math.abs(this._obj-this._obj_old) < NetworkConfig.objtol;
		
		this._version ++;
		
//		System.err.println("Word2count:");
//		Iterator<String> words = word2count.keySet().iterator();
//		while(words.hasNext()){
//			String word = words.next();
//			double count = word2count.get(word);
//			System.err.println(word+"\t"+count);
//		}
//		System.exit(1);
		
		return done;
	}
	
	//if the optimization seems to be done, it will return true.
	protected boolean updateDiscriminative(){
		
		this._opt.setVariables(this._weights);
    	this._opt.setObjective(-this._obj);
    	this._opt.setGradients(this._counts);
    	/*System.err.println("Global param");
    	System.err.println(Arrays.toString(this._weights));
    	System.err.println(-this._obj);
    	System.err.println(Arrays.toString(this._counts));
    	*/
//    	int fid = 10;
//    	System.err.println(Arrays.toString(this.getFeatureRep(fid))+"\t"+this._counts[fid]);
//    	System.exit(1);
    	
    	boolean done = false;
    	
    	try{
        	done = this._opt.optimize();
    	} catch(ExceptionWithIflag e){
    		//throw new NetworkException("Exception with Iflag:"+e.getMessage());
    		done = true;
    	}
		
    	if(Math.abs(this.getObj()-this.getObj_old())<NetworkConfig.objtol){
    		done = true;
    	}
    	
		this._version ++;
		return done;
	}
	
	public boolean isDiscriminative(){
		return this._isDiscriminative;
	}
	
	protected synchronized void resetCountsAndObj(){
		
		for(int k = 0 ; k<this._size; k++){
			this._counts[k] = 0.0;
			//for regularization
			if(this.isDiscriminative() && this._kappa > 0 && k>=this._fixedFeaturesSize){
				this._counts[k] += 2 * this._kappa * this._weights[k];
			}
		}
		this._obj = 0.0;
		//for regularization
		if(this.isDiscriminative() && this._kappa > 0){
			this._obj += - this._kappa * MathsVector.square(this._weights);
		}
		//NOTES:
		//for additional terms such as regularization terms:
		//always add to _obj the term g(x) you would like to maximize.
		//always add to _counts the NEGATION of the term g(x)'s gradient.
	}
	
	public boolean checkEqual(GlobalNetworkParam p){
		boolean v1 = Arrays.equals(this._weights, p._weights);
		boolean v2 = Arrays.deepEquals(this._feature2rep, p._feature2rep);
		return v1 && v2;
	}
	
	private void writeObject(ObjectOutputStream out)throws IOException{
		out.writeObject(this._featureIntMap);
		out.writeObject(this._feature2rep);
		out.writeObject(this._weights);
		out.writeInt(this._size);
		out.writeInt(this._fixedFeaturesSize);
		out.writeBoolean(this._locked);
	}
	
	@SuppressWarnings("unchecked")
	private void readObject(ObjectInputStream in)throws IOException, ClassNotFoundException{
		this._featureIntMap = (HashMap<String, HashMap<String, HashMap<String, Integer>>>)in.readObject();
		this._feature2rep = (String[][])in.readObject();
		this._weights = (double[])in.readObject();
		this._size = in.readInt();
		this._fixedFeaturesSize = in.readInt();
		this._locked = in.readBoolean();
	}

	public void setOptimzer(Optimizer opt) {
		this._opt = opt;
		
	}
	
}
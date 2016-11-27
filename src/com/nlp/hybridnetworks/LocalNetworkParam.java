/** Statistical Natural Language Processing System
    Copyright (C) 2014-2015  Lu, Wei

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

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;

//one thread should have one such LocalFeatureMap.
public class LocalNetworkParam implements Serializable{
	
	private static final long serialVersionUID = -2097104968915519992L;
	
	//the id of the thread.
	protected int _threadId;
	//the feature manager.
	protected FeatureManager _fm;
	//the version number.
	protected int _version;
	
	//the partial objective function.
	protected double _obj;
	//the local feature sparse array.
	protected int[] _fs;
	//the counts for all the features.
	protected double[] _counts;
	//mapping from global features to local features.
	protected HashMap<Integer, Integer> _globalFeature2LocalFeature;
	//check if it is finalized.
	protected boolean _isFinalized;
	
	//the cache that stores the features
	protected FeatureArray[][][] _cache;
	//check whether the cache is enabled.
	protected boolean _cacheEnabled = true;
	
	protected int _numNetworks;
	
	//if this is true, then we bypass the local params.
	protected boolean _globalMode;
	
	public LocalNetworkParam(int threadId, FeatureManager fm, int numNetworks){
		this._threadId = threadId;
		this._numNetworks = numNetworks;
		this._fm = fm;
		this._obj = 0.0;
		this._fs = null;
		//this gives you the mapping from global to local features.
		this._globalFeature2LocalFeature = new HashMap<Integer, Integer>();
		this._isFinalized = false;
		this._version = 0;
		this._globalMode = false;
		
		if(!NetworkConfig._CACHE_FEATURES_DURING_TRAINING){
			this.disableCache();
		}
		if(NetworkConfig._numThreads==1){
			this._globalMode = true;
		}

	}
	
	//when doing testing, we can set it to global model for improved efficiency.
	//the reason that we can do this global mode is because we don't update the params
	//during testing.
	public void setGlobalMode(){
		this._globalMode = true;
		this._globalFeature2LocalFeature = null;
	}
	
	//check whether it is in global mode.
	public boolean isGlobalMode(){
		return this._globalMode;
	}
	
	public int toLocalFeature(int f_global){
		if(this._globalMode){
			throw new RuntimeException("The current mode is global mode, converting a global feature to a local feature is not supported.");
		}
		//if it is not really a valid feature.
		if(f_global == -1){
			throw new RuntimeException("z");
//			return -1;
		}
		if(!this._globalFeature2LocalFeature.containsKey(f_global))
			this._globalFeature2LocalFeature.put(f_global, this._globalFeature2LocalFeature.size());
		return this._globalFeature2LocalFeature.get(f_global);
	}
	
	public int[] getFeatures(){
		return this._fs;
	}
	
	//get the version.
	public int getVersion(){
		return this._version;
	}
	
	//get the id of the thread.
	public int getThreadId(){
		return this._threadId;
	}
	
	//get the objective value.
	public double getObj(){
		return this._obj;
	}
	
	//add the objective value by a certain value.
	public void addObj(double obj){
		if(this._globalMode){
			this._fm.getParam_G().addObj(obj);
			return;
		}
		this._obj += obj;
	}
	
	//get the weight of a feature
	public double getWeight(int f){
		if(this.isGlobalMode()){
			return this._fm.getParam_G().getWeight(f);
		} else {
			try{
				return this._fm.getParam_G().getWeight(this._fs[f]);
			} catch(Exception e){
				throw new RuntimeException("fs is now:"+this._fs);
			}
		}
	}
	
	//the number of features.
	public int size(){
		return this._fs.length;
	}
	
	public void addCount(int f_local, double count){
//		if(this._globalMode){
//			throw new RuntimeException("The current mode is global mode, adding counts is not supported.");
//		}
		if(f_local == -1){
			throw new RuntimeException("x");
		}
		
		if(Double.isNaN(count)){
			throw new RuntimeException("NaN");
		}
		if(this._globalMode){
			this._fm.getParam_G().addCount(f_local, count);
			return;
		}
		this._counts[f_local] += count;
		
	}
	
	public double getCount(int f_local){
		if(this._globalMode){
			throw new RuntimeException("It's global mode, why do you do this?");
		}
		return this._counts[f_local];
	}
	
	public void reset(){
		if(this._globalMode){
			return;
		}
		this._obj = 0.0;
		Arrays.fill(this._counts, 0.0);
	}
	
	public void disableCache(){
		this._cache = null;
		this._cacheEnabled = false;
	}
	
	public boolean isCacheEnabled(){
		return this._cacheEnabled;
	}
	
	public FeatureArray extract(Network network, int parent_k, int[] children_k, int children_k_index){
		if(this.isCacheEnabled()){
			if(this._cache == null){
				this._cache = new FeatureArray[this._numNetworks][][];
			}
			if(this._cache[network.getNetworkId()] == null){
				this._cache[network.getNetworkId()] = new FeatureArray[network.countNodes()][];
			}
			if(this._cache[network.getNetworkId()][parent_k] == null){
				this._cache[network.getNetworkId()][parent_k] = new FeatureArray[network.getChildren(parent_k).length];
			}
			if(this._cache[network.getNetworkId()][parent_k][children_k_index] != null){
				return this._cache[network.getNetworkId()][parent_k][children_k_index];
			}
		}
		
		FeatureArray fa = this._fm.extract(network, parent_k, children_k, children_k_index);
		if(!this.isGlobalMode()){
			fa = fa.toLocal(this);
		}
		
		if(this.isCacheEnabled()){
			this._cache[network.getNetworkId()][parent_k][children_k_index] = fa;
		}
		
		return fa;
	}
	
	//finalize the param.
	public void finalizeIt(){
		//if it is global mode, do not have to do this at all.
		if(this.isGlobalMode()){
			System.err.println("Is global mode..");
			this._isFinalized = true;
			return;
		}
		this._fs = new int[this._globalFeature2LocalFeature.size()];
		Iterator<Integer> features = this._globalFeature2LocalFeature.keySet().iterator();
		while(features.hasNext()){
			int f_global = features.next();
			int f_local = this._globalFeature2LocalFeature.get(f_global);
			this._fs[f_local] = f_global;
		}
		if(NetworkConfig._CACHE_FEATURES_DURING_TRAINING){
			this._globalFeature2LocalFeature = null;
		}
		this._isFinalized = true;
		this._counts = new double[this._fs.length];
		System.err.println("Finalized local param. size:"+this._fs.length);
	}

}

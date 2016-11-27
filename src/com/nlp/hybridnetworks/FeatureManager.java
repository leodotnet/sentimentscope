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

import java.io.Serializable;
import java.util.Arrays;

public abstract class FeatureManager implements Serializable{
	
	private static final long serialVersionUID = 7999836838043433954L;
	
	//the number of networks.
	protected transient int _numNetworks;
	//the cache that stores the features
	protected transient FeatureArray[][][] _cache;
	
	//the parameters associated with the network.
	protected GlobalNetworkParam _param_g;
	//the local feature maps, one for each thread.
	protected LocalNetworkParam[] _params_l;
	//check whether the cache is enabled.
	protected boolean _cacheEnabled = false;
	
	protected int _numThreads;
	
	public FeatureManager(GlobalNetworkParam param_g){
		this._param_g = param_g;
		this._numThreads = NetworkConfig._numThreads;
		this._params_l = new LocalNetworkParam[this._numThreads];
		this._cacheEnabled = false;
	}
	
	public void setLocalNetworkParams(int threadId, LocalNetworkParam param_l){
		this._params_l[threadId] = param_l;
	}
	
	//update the parameters.
	public synchronized boolean update(){
		
		//if the number of thread is 1, then your local param fetches information directly from the global param.
		if(NetworkConfig._numThreads!=1){
			this._param_g.resetCountsAndObj();
			
			for(LocalNetworkParam param_l : this._params_l){
				int[] fs = param_l.getFeatures();
				for(int f_local = 0; f_local<fs.length; f_local++){
					int f_global = fs[f_local];
					double count = param_l.getCount(f_local);
					this._param_g.addCount(f_global, count);
				}
				this._param_g.addObj(param_l.getObj());
			}
		}
		
		boolean done = this._param_g.update();
		
		//System.err.println("count:"+Arrays.toString(getParam_G()._counts));
		//System.err.println("weight: "+Arrays.toString(getParam_G()._weights));
		
		if(NetworkConfig._numThreads != 1){
			for(LocalNetworkParam param_l : this._params_l){
				param_l.reset();
			}
		} else {
			this._param_g.resetCountsAndObj();
		}
		return done;
	}
	
	public void enableCache(int numNetworks){
		this._numNetworks = numNetworks;
		this._cache = new FeatureArray[numNetworks][][];
		this._cacheEnabled = true;
	}
	
	public void disableCache(){
		this._cache = null;
		this._cacheEnabled = false;
	}
	
	public boolean isCacheEnabled(){
		return this._cacheEnabled;
	}
	
	public GlobalNetworkParam getParam_G(){
		return this._param_g;
	}
	
	public LocalNetworkParam[] getParams_L(){
		return this._params_l;
	}
	
	public FeatureArray extract(Network network, int parent_k, int[] children_k, int children_k_index){
		if(this.isCacheEnabled()){
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
		
		FeatureArray fa = this.extract_helper(network, parent_k, children_k);
		
		if(this.isCacheEnabled()){
			this._cache[network.getNetworkId()][parent_k][children_k_index] = fa;
		}
		return fa;
	}
	
	protected abstract FeatureArray extract_helper(Network network, int parent_k, int[] children_k);
	
}
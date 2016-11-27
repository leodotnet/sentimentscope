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

import com.nlp.commons.types.Instance;

public class LocalNetworkLearnerThread extends Thread {
	
	//the id of the thread.
	private int _threadId = -1;
	
	//the max number of nodes in the network.
	private int _networkCapacity = 1000000;
	//the local feature map.
	private LocalNetworkParam _param;
	
	//check whether we cache the networks.
//	private boolean _cacheNetworks = true;
	private boolean _cacheNetworks = true;
	//the networks.
	private Network[] _networks;
	
	private Instance[] _instances;
	private NetworkCompiler _builder;
	private int _it;
	
	public LocalNetworkLearnerThread copyThread(){
		if(this._cacheNetworks){
			return new LocalNetworkLearnerThread(this._threadId, this._param, this._instances, this._networks, this._it+1);
		} else {
			return new LocalNetworkLearnerThread(this._threadId, this._param, this._instances, this._builder, this._it+1);
		}
	}
	
	private LocalNetworkLearnerThread(int threadId, LocalNetworkParam param, Instance[] instances, NetworkCompiler builder, int it){
		this._threadId = threadId;
		this._param = param;
		this._instances = instances;
		this._builder = builder;
		this._it = it;
	}
	
	private LocalNetworkLearnerThread(int threadId, LocalNetworkParam param, Instance[] instances, Network[] networks, int it){
		this._threadId = threadId;
		this._param = param;
		this._instances = instances;
		this._networks = networks;
		this._it = it;
	}
	
	//please make sure the threadId is 0-indexed.
	public LocalNetworkLearnerThread(int threadId, FeatureManager fm, Instance[] instances, NetworkCompiler builder, int it){
		this._threadId = threadId;
		this._param = new LocalNetworkParam(this._threadId, fm, instances.length);
		fm.setLocalNetworkParams(this._threadId, this._param);
		
		this._builder = builder;
		this._instances = instances;
		
		if(this._cacheNetworks)
			this._networks = new Network[this._instances.length];
		
		this._it = it;
	}
	
	public int getThreadId(){
		return this._threadId;
	}
	
    @Override
    public void run() {
    	this.train(this._it);
    }
    
	public void touch(){
		
		//for now, disable the feature cache...
//		this._param.disableCache();
		
		long time = System.currentTimeMillis();
		//extract the features..
		for(int networkId = 0; networkId< this._instances.length; networkId++){
			if(networkId%100==0)
				System.err.print('.');
			this.getNetwork(networkId).touch();
		}
		System.err.println();
		time = System.currentTimeMillis() - time;
		System.out.println("Thread "+this._threadId + " touch time: "+ time/1000.0+" secs.");
		
		this._param.finalizeIt();
		
	}
	
	private void train(int it){
		
		for(int networkId = 0; networkId< this._instances.length; networkId++){
			Network network = this.getNetwork(networkId);
			network.train();
		}
		
	}
	
	private Network getNetwork(int networkId){
		if(this._cacheNetworks && this._networks[networkId]!=null)
			return this._networks[networkId];
		Network network = this._builder.compile(networkId, this._instances[networkId], this._param);
		if(this._cacheNetworks)
			this._networks[networkId] = network;
		if(network.countNodes() > this._networkCapacity) this._networkCapacity = network.countNodes();
		return network;
	}
	
	public int getNetworkCapacity(){
		return this._networkCapacity;
	}
	
	public LocalNetworkParam getLocalNetworkParam(){
		return this._param;
	}
	
	public Network[] getNetworks()
	{
		return this._networks;
	}
}
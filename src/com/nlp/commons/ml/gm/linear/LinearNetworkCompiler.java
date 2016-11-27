package com.nlp.commons.ml.gm.linear;

import com.nlp.commons.types.Instance;
import com.nlp.commons.types.OutputToken;
import com.nlp.hybridnetworks.LocalNetworkParam;
import com.nlp.hybridnetworks.Network;
import com.nlp.hybridnetworks.NetworkCompiler;
import com.nlp.hybridnetworks.NetworkIDMapper;

public class LinearNetworkCompiler extends NetworkCompiler{
	
	private static final long serialVersionUID = 4344621495273642001L;

	public enum nodeType {LEAF, MIDDLE, ROOT};
	
	private OutputToken[] _allOutputs;
	
	public LinearNetworkCompiler(OutputToken[] allOutputs) {
		this._allOutputs = allOutputs;
	}
	
	@Override
	public LinearInstance decompile(Network network) {
		LinearNetwork linearNetwork = (LinearNetwork)network;
		LinearInstance inst = (LinearInstance)linearNetwork.getInstance();
		OutputToken[] prediction = new OutputToken[inst.size()];
		
		int k = prediction.length-1;
		int node_k = linearNetwork.countNodes()-1;
		while(true){
			node_k = linearNetwork.getMaxPath(node_k)[0];
			int[] ids = NetworkIDMapper.toHybridNodeArray(linearNetwork.getNode(node_k));
			if(ids[4]==nodeType.MIDDLE.ordinal()){
				prediction[k--] = this._allOutputs[ids[3]];
			} else if(ids[4]==nodeType.LEAF.ordinal()){
				break;
			}
		}
		if(k!=-1){
			throw new RuntimeException("k="+k);
		}
		
//		int node_k = linearNetwork.countNodes()-1;
//		for(int k = prediction.length-1; k>=0; k--){
//			node_k = linearNetwork.getMaxPath(node_k)[0];
////			System.err.println(node_k);
//			int[] ids = NetworkIDMapper.toHybridNodeArray(linearNetwork.get(node_k));
//			prediction[k] = this._allOutputs[ids[3]];
////			System.err.println(k+"=>"+node_k+"\t"+ids[3]+"\t"+prediction[k]+"\t"+inst.getOutput()[k]);
//		}
		
		inst.setPrediction(prediction);
		
		return inst;
	}
	
	@Override
	public LinearNetwork compile(int networkId, Instance inst, LocalNetworkParam param) {
		if(inst.isLabeled()){
			return this.compileLabeled(networkId, (LinearInstance)inst, param);
		} else {
			return this.compileUnlabeled(networkId, (LinearInstance)inst, param);
		}
	}
	
	private LinearNetwork compileLabeled(int networkId, LinearInstance inst, LocalNetworkParam param){
		
		LinearNetwork network = new LinearNetwork(networkId, inst, param);
		
		OutputToken[] outputs = inst.getOutput();
		
		long leaf = this.toNode_leaf();
		network.addNode(leaf);
		
		long[] children;
		
		children = new long[]{leaf};
		for(int pos = 0; pos < inst.size(); pos++){
			long node = this.toNode(pos, outputs[pos]);
			network.addNode(node);
			network.addEdge(node, children);
			children = new long[]{node};
		}
		
		long root = this.toNode_root(inst.size());
		network.addNode(root);
		network.addEdge(root, children);
		
		network.finalizeNetwork();
		
//		System.err.println(network.countNodes()+" nodes.");
//		System.exit(1);
		
		return network;
		
	}
	
	//the cache.
	private long[] nodes_cache;
	private int[][][] children_cache;
	
	private int _maxLen_cache = -1;
	
	private void compileUnlabeled_cache(int len){
		
		if(len > this._maxLen_cache){
			System.err.println("Okay, update the len to "+len);
			this._maxLen_cache = len;
		} else {
			return;
		}
		
		LinearNetwork network = new LinearNetwork();
		
		long leaf = this.toNode_leaf();
		network.addNode(leaf);
		
		long[] children;
		
		children = new long[]{leaf};
		for(int pos = 0; pos < this._maxLen_cache; pos++){
			long[] children_new = new long[this._allOutputs.length];
			
			for(int k = 0; k<this._allOutputs.length; k++){
				long node = this.toNode(pos, this._allOutputs[k]);
				children_new[k] = node;
				network.addNode(node);
				for(long child : children)
					network.addEdge(node, new long[]{child});
			}
			
			long root = this.toNode_root(pos);
			network.addNode(root);
			for(long child : children)
				network.addEdge(root, new long[]{child});
			
			children = children_new;
		}
		
		long stop = this.toNode_root(this._maxLen_cache);
		network.addNode(stop);
		for(long child : children)
			network.addEdge(stop, new long[]{child});
		
		network.finalizeNetwork();
		
		this.nodes_cache = network.getAllNodes();
		this.children_cache = network.getAllChildren();
	}
	
	private boolean cache_unlabeled_network = true;
	
	private int countNodes(LinearInstance inst){
		return 1 + inst.size() * (this._allOutputs.length + 1) + 1;
	}
	
	private LinearNetwork compileUnlabeled(int networkId, LinearInstance inst, LocalNetworkParam param){
		
		if(this.cache_unlabeled_network){
			this.compileUnlabeled_cache(inst.size());
			FasterLinearNetwork network =  new FasterLinearNetwork(networkId, inst, this.nodes_cache, this.children_cache, param, this.countNodes(inst));
//			System.err.println(".."+nodes_cache.length);
			int v = NetworkIDMapper.toHybridNodeArray(this.nodes_cache[this.countNodes(inst)-1])[4];
//			System.err.println("v="+v+"\t"+nodeType.ROOT.ordinal());
			if(v!=nodeType.ROOT.ordinal()){
				throw new RuntimeException("v value mismatch:"+v+"\t"+nodeType.ROOT.ordinal());
			}
			return network;
		}
		
		LinearNetwork network = new LinearNetwork(networkId, inst, param);
		
		long start = this.toNode_leaf();
		network.addNode(start);
		
		long[] children;
		
		children = new long[]{start};
		
		for(int pos = 0; pos < inst.size(); pos++){
			long[] children_new = new long[this._allOutputs.length];
			
			//the first two are START and FINISH
			for(int k = 0; k<this._allOutputs.length; k++){
				long node = this.toNode(pos, this._allOutputs[k]);
				children_new[k] = node;
				network.addNode(node);
//				System.err.println(node+"\t"+Arrays.toString(children)+"\t"+k+"\t"+this._allOutputs[k].getId()+"\t"+this._allOutputs[k]);
				for(long child : children)
					network.addEdge(node, new long[]{child});
			}
			
			long stop = this.toNode_root(pos);
			network.addNode(stop);
			for(long child : children)
				network.addEdge(stop, new long[]{child});
			
			children = children_new;
		}
		
		long stop = this.toNode_root(inst.size());
		network.addNode(stop);
		for(long child : children)
			network.addEdge(stop, new long[]{child});
		
		network.finalizeNetwork();
		
		return network;
		
	}
	
	private long toNode_leaf(){
		return NetworkIDMapper.toHybridNodeID(new int[]{0, 0, 0, 0, nodeType.LEAF.ordinal()});
	}
	
	private long toNode_root(int pos){
		return NetworkIDMapper.toHybridNodeID(new int[]{pos+1, pos, 0, 0, nodeType.ROOT.ordinal()});
	}
	
	private long toNode(int pos, OutputToken tag){
		return NetworkIDMapper.toHybridNodeID(new int[]{pos+1, pos, pos+1, tag.getId(), nodeType.MIDDLE.ordinal()});
	}

}
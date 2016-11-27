package com.nlp.commons.ml.gm.linear;

import java.util.ArrayList;
import java.util.Collections;

import com.nlp.commons.types.Instance;
import com.nlp.commons.types.OutputToken;
import com.nlp.hybridnetworks.LocalNetworkParam;
import com.nlp.hybridnetworks.Network;
import com.nlp.hybridnetworks.NetworkCompiler;
import com.nlp.hybridnetworks.NetworkConfig;
import com.nlp.hybridnetworks.NetworkIDMapper;

public class LinearSpanNetworkCompiler extends NetworkCompiler{
	
	private static final long serialVersionUID = 4344621495273642001L;
	
	public enum nodeType {LEAF, SPAN, END, ROOT};
	
	private OutputToken[] _allOutputs;

	//the cache.
	private long[] nodes_cache;
	private int[][][] children_cache;
	
	private int _maxLen_cache = -1;
	private int _maxSpanLen = NetworkConfig._maxSpanLen;
	
	public LinearSpanNetworkCompiler(OutputToken[] allOutputs) {
		this._allOutputs = allOutputs;
	}
	
	@Override
	public LinearSpanInstance decompile(Network network) {
		LinearSpanNetwork linearNetwork = (LinearSpanNetwork)network;
		LinearSpanInstance inst = (LinearSpanInstance)linearNetwork.getInstance();
		ArrayList<LinearSpan> preds = new ArrayList<LinearSpan>();
		
		int node_k = linearNetwork.countNodes()-1;
		while(true){
			node_k = linearNetwork.getMaxPath(node_k)[0];
			int[] ids = NetworkIDMapper.toHybridNodeArray(linearNetwork.getNode(node_k));
			if(ids[4] == nodeType.SPAN.ordinal()){
				int bIndex = ids[1];
				int eIndex = ids[0];
				OutputToken tag = this._allOutputs[ids[3]];
				LinearSpan span = new LinearSpan(bIndex, eIndex, tag);
				preds.add(span);
			}
			if(ids[4] == nodeType.LEAF.ordinal()){
				break;
			}
		}
		
		Collections.sort(preds);
		
//		for(int k = 0; k<preds.size(); k++){
//			LinearSpan pred = preds.get(k);
//			if(pred.getOutputToken().getName().endsWith("-B")){
//				int bIndex = pred.getBIndex();
//				String name_orig = pred.getOutputToken().getName();
//				String name = name_orig.substring(0, name_orig.length()-2);
//				OutputTag tag = LinearSpanInstanceReader.toOutputTag(name);
//				preds.remove(k);
//				
//				while(true){
//					LinearSpan pred_next = preds.get(k);
//					if(pred_next.getOutputToken().getName().equals(name+"-L")){
//						int eIndex = pred_next.getEIndex();
//						LinearSpan span_new = new LinearSpan(bIndex, eIndex, tag);
//						preds.add(k, span_new);
//						break;
//					} else if(!pred_next.getOutputToken().getName().equals(name+"-I")){
//						throw new RuntimeException("The tag is invalid:"+pred_next.getOutputToken().getName()+"!="+name+"-I");
//					} else {
//						preds.remove(k);
//					}
//				}
//			}
//			if(pred.getOutputToken().getName().endsWith("-U")){
//				throw new RuntimeException("This is invalid:"+pred.getOutputToken().getName());
//			}
//		}
		
		LinearSpan[] prediction = new LinearSpan[preds.size()];
		for(int k = 0; k<prediction.length; k++){
			prediction[k] = preds.get(k);
		}
		
		inst.setPrediction(prediction);
		
		return inst;
	}
	
	@Override
	public LinearSpanNetwork compile(int networkId, Instance inst, LocalNetworkParam param) {
		if(inst.isLabeled())
			return this.compileLabeled(networkId, (LinearSpanInstance)inst, param);
		else
			return this.compileUnlabeled(networkId, (LinearSpanInstance)inst, param);
	}
	
	private LinearSpanNetwork compileLabeled(int networkId, LinearSpanInstance inst, LocalNetworkParam param){
		
		LinearSpanNetwork network = new LinearSpanNetwork(networkId, inst, param);
		
		LinearSpan[] spans = (LinearSpan[])inst.getOutput();
		
		long leaf = this.toNode_leaf();
		network.addNode(leaf);
		
		long[] children;
		
		children = new long[]{leaf};
		
		for(LinearSpan span : spans){
			int bIndex = span.getBIndex();
			int eIndex = span.getEIndex();
			OutputToken tag = span.getOutputToken();
			
//			System.err.println(bIndex+"\t"+eIndex+"\t"+tag);
			long node_span = this.toNode_span(bIndex, eIndex, tag);
			network.addNode(node_span);
			network.addEdge(node_span, children);
			
			long node_end = this.toNode_end(eIndex, tag);
			network.addNode(node_end);
			network.addEdge(node_end, new long[]{node_span});
			
			children = new long[]{node_end};
		}
		
		long root = this.toNode_root(inst.size());
		network.addNode(root);
		network.addEdge(root, children);
		
		network.finalizeNetwork();
		
		return network;
	}
	
	private void compileUnlabeled_cache(int len){
		
		if(len > this._maxLen_cache){
			System.err.println("Okay, update the len to "+len);
			this._maxLen_cache = len;
		} else {
			return;
		}
		
		LinearNetwork network = new LinearNetwork();
		
		long node_leaf = this.toNode_leaf();
		network.addNode(node_leaf);
		
		long node_root = this.toNode_root(0);
		network.addNode(node_root);
		network.addEdge(node_root, new long[]{node_leaf});
		
		for(int eIndex = 1; eIndex <= this._maxLen_cache; eIndex++){
			node_root = this.toNode_root(eIndex);
			network.addNode(node_root);
			for(int k = 0; k<this._allOutputs.length; k++){
				OutputToken output = this._allOutputs[k];
				long node_end = this.toNode_end(eIndex, output);
				network.addNode(node_end);
				int size = Math.min(this._maxSpanLen, eIndex);
				for(int L = 1; L<=  size; L++){
					int bIndex = eIndex - L;
					long node_span = this.toNode_span(bIndex, eIndex, output);
					network.addNode(node_span);
					network.addEdge(node_end, new long[]{node_span});
					if(bIndex==0){
						network.addEdge(node_span, new long[]{node_leaf});
					} else {
						for(int j = 0; j<this._allOutputs.length; j++){
							long node_end_prev = this.toNode_end(bIndex, this._allOutputs[j]);
							network.addEdge(node_span, new long[]{node_end_prev});
						}
					}
				}
				network.addEdge(node_root, new long[]{node_end});
			}
		}
		
		network.finalizeNetwork();
		
		this.nodes_cache = network.getAllNodes();
		this.children_cache = network.getAllChildren();
		
		//checking..
		int exp_size = this.countNodes(len);
		int act_size = this.nodes_cache.length;
		if(exp_size!=act_size){
			throw new RuntimeException("size mismatch:"+exp_size+"!="+act_size);
		} else {
			System.err.println("size match:"+exp_size+"="+act_size);
		}
	}
	
	private boolean cache_unlabeled_network = true;
	
	private int countNodes(int inst_size){
		
//		System.err.println(this._allOutputs.length+"\t"+inst.size()+"\t"+Arrays.toString(this._allOutputs));
		
		return 1 + (this._allOutputs.length) * inst_size + (inst_size + 1) 
				+ (this._allOutputs.length) * this.countSpans(inst_size);
				//leaf + end + root + span
	}
	
	private int countNodes(LinearSpanInstance inst){
		
//		System.err.println(this._allOutputs.length+"\t"+inst.size()+"\t"+Arrays.toString(this._allOutputs));
//		System.exit(1);
		
		return 1 + (this._allOutputs.length) * inst.size() + (inst.size() + 1) 
				+ (this._allOutputs.length) * this.countSpans(inst.size());
				//leaf + end + root + span
	}
	
	private int countSpans(int len){
		if(len<=NetworkConfig._maxSpanLen){
			return (1+len)*len/2;
		}
		int prefix = (1+NetworkConfig._maxSpanLen)*NetworkConfig._maxSpanLen/2;
		int suffix = (len-NetworkConfig._maxSpanLen)*NetworkConfig._maxSpanLen;
		return prefix + suffix;
	}
	
	private synchronized LinearSpanNetwork compileUnlabeled(int networkId, LinearSpanInstance inst, LocalNetworkParam param){
		
		if(this.cache_unlabeled_network){
			this.compileUnlabeled_cache(inst.size());
			FasterLinearSpanNetwork network =  new FasterLinearSpanNetwork(networkId, inst, this.nodes_cache, this.children_cache, param, this.countNodes(inst));
			try{
				int v = NetworkIDMapper.toHybridNodeArray(this.nodes_cache[this.countNodes(inst)-1])[4];
				if(v!=nodeType.ROOT.ordinal()){
					throw new RuntimeException("v="+v+"\t"+nodeType.ROOT.ordinal());
				}
			} catch(Exception e){
				System.err.println("actual:"+nodes_cache.length);
				System.err.println("look 4:"+this.countNodes(inst));
			}
			return network;
		}
		
		LinearSpanNetwork network = new LinearSpanNetwork(networkId, inst, param);
		
		long node_leaf = this.toNode_leaf();
		network.addNode(node_leaf);
		
		long node_root = this.toNode_root(0);
		network.addNode(node_root);
		network.addEdge(node_root, new long[]{node_leaf});
		
		for(int eIndex = 1; eIndex <= inst.size(); eIndex++){
			node_root = this.toNode_root(eIndex);
			network.addNode(node_root);
			for(int k = 0; k<this._allOutputs.length; k++){
				OutputToken output = this._allOutputs[k];
				long node_end = this.toNode_end(eIndex, output);
				network.addNode(node_end);
				int size = Math.min(this._maxSpanLen, eIndex);
				for(int L = 1; L<=  size; L++){
					int bIndex = eIndex - L;
					long node_span = this.toNode_span(bIndex, eIndex, output);
					network.addNode(node_span);
					network.addEdge(node_end, new long[]{node_span});
					if(bIndex==0){
						network.addEdge(node_span, new long[]{node_leaf});
					} else {
						for(int j = 0; j<this._allOutputs.length; j++){
							long node_end_prev = this.toNode_end(bIndex, this._allOutputs[j]);
							network.addEdge(node_span, new long[]{node_end_prev});
						}
					}
				}
				network.addEdge(node_root, new long[]{node_end});
			}
		}
		
		network.finalizeNetwork();
		
		return network;
	}
	
	private long toNode_leaf(){
		return NetworkIDMapper.toHybridNodeID(new int[]{0, 0, 0, 0, nodeType.LEAF.ordinal()});
	}
	
	private long toNode_root(int eIndex){
		return NetworkIDMapper.toHybridNodeID(new int[]{eIndex, eIndex, 1, 0, nodeType.ROOT.ordinal()});
	}
	
	private long toNode_span(int bIndex, int eIndex, OutputToken tag){
		return NetworkIDMapper.toHybridNodeID(new int[]{eIndex, bIndex, 0, tag.getId(), nodeType.SPAN.ordinal()});
	}
	
	private long toNode_end(int eIndex, OutputToken tag){
		return NetworkIDMapper.toHybridNodeID(new int[]{eIndex, eIndex, 0, tag.getId(), nodeType.END.ordinal()});
	}

}
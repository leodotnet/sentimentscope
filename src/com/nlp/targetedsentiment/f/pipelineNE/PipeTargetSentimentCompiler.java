package com.nlp.targetedsentiment.f.pipelineNE;



import java.util.ArrayList;

import com.nlp.commons.ml.gm.linear.LinearInstance;
import com.nlp.commons.ml.gm.linear.LinearNetwork;
import com.nlp.commons.ml.gm.linear.LinearNetworkCompiler;
import com.nlp.commons.ml.gm.linear.OutputTag;
import com.nlp.commons.types.InputToken;
import com.nlp.commons.types.Instance;
import com.nlp.commons.types.OutputToken;
import com.nlp.hybridnetworks.LocalNetworkParam;
import com.nlp.hybridnetworks.Network;
import com.nlp.hybridnetworks.NetworkIDMapper;
import com.nlp.targetedsentiment.util.WordWithFeatureToken;




public class PipeTargetSentimentCompiler extends LinearNetworkCompiler {

	int NEMaxLength = 3;
	int SpanMaxLength = 10;
	boolean full_connected = false;
	/*
	TargetSentimentViewer viewer = new TargetSentimentViewer(this, null, 5);*/
	public static boolean visual = true;
	
	
	public PipeTargetSentimentCompiler() {
		super(null);
		// TODO Auto-generated constructor stub
	}
	
	public PipeTargetSentimentCompiler(int NEMaxLength, int SpanMaxLength) {
		super(null);
		this.NEMaxLength = NEMaxLength;
		this.SpanMaxLength = SpanMaxLength;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 2100499563741744475L;
	
	public enum NodeType {Start, Span, End};
	
	public enum SubNodeType {O, B, I}
	
	//public enum SentType {positive, negative, neutral}
	
	private OutputToken[] _allOutputs;
	
	PipeTargetSentimentFeatureManager fm;

	

	
	int SubNodeTypeSize = SubNodeType.values().length;
	
	

	
	public void setOutputToken(OutputToken[] allOutputs)
	{
		this._allOutputs = allOutputs;
	}
	
	public void setFeatureManager(PipeTargetSentimentFeatureManager fm)
	{
		this.fm = fm;
	}
	
	@Override
	public LinearInstance decompile(Network network) {
		//Network network = (Network)network;
		LinearInstance inst = (LinearInstance)network.getInstance();
		int size = inst.size();
		InputToken[] input = inst.getInput();
	
		ArrayList<int[]> preds = new ArrayList<int[]>();
		ArrayList<int[]> preds_refine = new ArrayList<int[]>();

		
		int node_k = network.countNodes()-1;
		//node_k = 0;
		while(true){
			//System.out.print(node_k + " ");
			node_k = network.getMaxPath(node_k)[0];
			
			int[] ids = NetworkIDMapper.toHybridNodeArray(network.getNode(node_k));
			//System.out.println("ids:" + Arrays.toString(ids));
			
			if (ids[4] == NodeType.End.ordinal())
			{
				break;
			}
			
			int pos = size - ids[0];
			int subnode = SubNodeTypeSize - ids[1];
			int node_type = ids[4];
			
			if(ids[4] == NodeType.Span.ordinal()){
				
				preds.add(new int[]{pos, subnode});
				
			}
			
		}
		

		
		ArrayList<OutputTag> predication_array = new ArrayList<OutputTag>();
		String subnode = "";

		
		for(int i = 0; i < preds.size(); i++)
		{
			int[] ids = preds.get(i);
			int pos = ids[0];
			int subnode_index = ids[1];

			subnode = SubNodeType.values()[subnode_index].name();
			//subnode = subnode.replace('_', '-');
			
			predication_array.add(new OutputTag(subnode));	
			

		}
		
		//System.out.println();

		
	
		OutputTag[] prediction = new OutputTag[size];
		
		//System.out.println("\n~~\n");
		for(int k = 0; k < prediction.length; k++)
		{
			prediction[k] = predication_array.get(k);
			//System.out.print(prediction[k].getName() + " ");
		}
		//System.out.println();
		
		
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
	
	private LinearNetwork compileUnlabeled(int networkId, LinearInstance inst, LocalNetworkParam param){
		LinearNetwork network = new LinearNetwork(networkId, inst, param);
		
		InputToken[] inputs = inst.getInput();
		
		WordWithFeatureToken[] my_inputs = (WordWithFeatureToken[] )inputs;

		//OutputToken[] outputs = inst.getOutput();
		
		int size = inst.size();
		

		
		long start = this.toNode_start(size);
		network.addNode(start);
		
		long[][] node_array = new long[size][SubNodeTypeSize];
		
		//build node array
		for(int pos = 0; pos < size; pos++)
		{
			for(int subtype = 0; subtype < SubNodeTypeSize; subtype++)
			{
				
				{
					long node = this.toNode_Span(size, pos, subtype);
					network.addNode(node);
					node_array[pos][subtype] = node;
				}
			}
		}
		
		long end = this.toNode_end(inst.size());
		network.addNode(end);
		

		long from = -1, to = -1;

		
		if (full_connected)
		{
			//Start to B & O
			for(int i = 0; i < SubNodeTypeSize; i++)
			{
				from = start;
				to = node_array[0][i];
				network.addEdge(from, new long[]{to});
			}
			
			//sentence	
			for(int pos = 0; pos < size - 1 ; pos++)
			{
				for(int i = 0; i < SubNodeTypeSize; i++)
				{
					for(int j = 0; j < SubNodeTypeSize; j++)
					{
						from = node_array[pos][i];
						to = node_array[pos + 1][j];
						network.addEdge(from, new long[]{to});
					}
				}
				
			}
			
			
			//add last column of span node to end
			for(int i = 0; i < SubNodeTypeSize; i++)
			{
				from = node_array[size - 1][i];
				to = end;
				network.addEdge(from, new long[]{to});
				
			}
			
			
			network.finalizeNetwork();
			
			/*
			if (visual)
				viewer.visualizeNetwork(network, null, "Sentiment Model:unlabeled[" + networkId + "]");
			*/
			return network;
			
		}
		
		
		
		//Start to B & O
		for(int i = 0; i < SubNodeTypeSize - 1; i++)
		{
			from = start;
			to = node_array[0][i];
			network.addEdge(from, new long[]{to});
		}
		
		//sentence	
		for(int pos = 0; pos < size - 1 ; pos++)
		{
			
			//current is O, next is O or B
			for(int j = 0; j < SubNodeTypeSize - 1; j++)
			{
				from = node_array[pos][0];
				to = node_array[pos + 1][j];
				network.addEdge(from, new long[]{to});
			}
			
			//current is B, next is O, B, I
			for(int i = 1; i < SubNodeTypeSize; i++)
			{
				for(int j = 0; j < SubNodeTypeSize; j++)
				{
					from = node_array[pos][i];
					to = node_array[pos + 1][j];
					network.addEdge(from, new long[]{to});
				}
			}
			
		}
		
		
		//add last column of span node to end
		for(int i = 0; i < SubNodeTypeSize; i++)
		{
			from = node_array[size - 1][i];
			to = end;
			network.addEdge(from, new long[]{to});
			
		}
		
		
		network.finalizeNetwork();
		
		/*
		if (visual)
			viewer.visualizeNetwork(network, null, "Sentiment Model:unlabeled[" + networkId + "]");*/
		
		
		
		
		
		return network;
	}

	private LinearNetwork compileLabeled(int networkId, LinearInstance inst, LocalNetworkParam param){
		
		LinearNetwork network = new LinearNetwork(networkId, inst, param);
		
		WordWithFeatureToken[] inputs = (WordWithFeatureToken[])inst.getInput();
		
		OutputToken[] outputs = inst.getOutput();
			
		int size = inst.size();
		
		long start = this.toNode_start(size);
		network.addNode(start);
		
		long[][] node_array = new long[size][SubNodeTypeSize];
		
		//build node array
		for(int pos = 0; pos < size; pos++)
		{
			for(int subtype = 0; subtype < SubNodeTypeSize; subtype++)
			{
				long node = this.toNode_Span(size, pos, subtype);
				network.addNode(node);
				node_array[pos][subtype] = node;
				
			}
		}
		
		long end = this.toNode_end(inst.size());
		network.addNode(end);
		
		
		////////////////////////////////////////////
		
		
		int last_entity_pos = -1;
		SubNodeType last_polar = null;
		SubNodeType subnode = null;
		long from = -1;
		long to = -1;
		int entity_begin = -1;
		
		

		long last_one = start;
		
		for(int pos = 0; pos < size; pos++)
		{
			String word = inputs[pos].getName();
			String label = outputs[pos].getName();
			
			label = label.substring(0, 1);
			subnode = SubNodeType.valueOf(label);
		
			from = last_one;
			to = node_array[pos][subnode.ordinal()];
			network.addEdge(from,  new long[]{to});
			
			last_one = to;
		}
		
		
		//add the last column node to end
		if (subnode != null)
		{
			from = last_one;
			to = end;
			network.addEdge(from,  new long[]{to});
		} else {
			//polar = SentType._;
			//from = start;
			//to = node_array[0][polar.ordinal()][SubNodeType.B.ordinal()];
			//network.addEdge(from,  new long[]{to});
			
			System.out.println("No Entity found in this Instance, Discard!");
			/*
			for(int pos = 0; pos < size; pos++)
			{
				
			}*/
			
		}
		
		
		
		network.finalizeNetwork();
		/*
		if (visual)
		viewer.visualizeNetwork(network, null, "Sentiment Model:labeled[" + networkId + "]");*/
//		System.err.println(network.countNodes()+" nodes.");
//		System.exit(1);
		
		return network;
		
	}
	
	boolean startOfEntity(int pos, int size, WordWithFeatureToken[] inputs)
	{
		String label = inputs[pos].getName();
		if (label.startsWith("B"))
			return true;
		
		if (pos == 0 && label.startsWith("I"))
			return true;
		
		if (pos > 0)
		{
			String prev_label =  inputs[pos - 1].getName();
			if (label.startsWith("I") && prev_label.startsWith("O"))
				return true;
		}
		
		
		return false;
	}
	
	boolean endofEntity(int pos, int size, OutputToken[] outputs)
	{
		String label = outputs[pos].getName();
		if (!label.startsWith("O"))
		{
			if (pos == size - 1)
				return true;
			else {
				String next_label =  outputs[pos + 1].getName();
				if (next_label.startsWith("O") || next_label.startsWith("B"))
					return true;
			}
		}
		
		return false;
	}
	
	
	
	private long toNode_start(int size){
		return NetworkIDMapper.toHybridNodeID(new int[]{size + 1, 0, 0, 0, NodeType.Start.ordinal()});
	}
	
	
//	private long toNode_hiddenState(int size, int bIndex, OutputToken hiddenState){
//		//System.out.println("bIndex=" + bIndex);
//		return NetworkIDMapper.toHybridNodeID(new int[]{size - bIndex, 2, 0, hiddenState.getId(), nodeType.HiddenState.ordinal()});
//	}
	
	//private long toNode_Entity(int size, int bIndex, int row, EntityNodeType type){
		//System.out.println("bIndex=" + bIndex);
	//	return NetworkIDMapper.toHybridNodeID(new int[]{size - bIndex, NodeTypeSize - row, 0, 0, nodeType.Entity.ordinal()});
	//}
	
	private long toNode_Span(int size, int bIndex, int subnode){
		//System.out.println("bIndex=" + bIndex);
		return NetworkIDMapper.toHybridNodeID(new int[]{size - bIndex, SubNodeTypeSize - subnode, 0, 0, NodeType.Span.ordinal()});
	}
	
	
//	private long toNode_observation(int size, int bIndex, InputToken observation){
//		return NetworkIDMapper.toHybridNodeID(new int[]{size - bIndex, 1, 0, observation.getId(), nodeType.Observation.ordinal()});
//	}
//	
//	
	
	
	/**/
	private long toNode_end(int size){
		return NetworkIDMapper.toHybridNodeID(new int[]{0, 0, 0, 0, NodeType.End.ordinal()});
	}
	
	
	
	
	

}

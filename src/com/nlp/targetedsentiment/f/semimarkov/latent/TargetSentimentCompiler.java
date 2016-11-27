package com.nlp.targetedsentiment.f.semimarkov.latent;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;

import com.nlp.commons.ml.gm.linear.LinearInstance;
import com.nlp.commons.ml.gm.linear.LinearNetwork;
import com.nlp.commons.ml.gm.linear.LinearNetworkCompiler;
import com.nlp.commons.ml.gm.linear.LinearSpan;
import com.nlp.commons.ml.gm.linear.LinearSpanInstance;
import com.nlp.commons.ml.gm.linear.LinearSpanNetwork;
import com.nlp.commons.ml.gm.linear.OutputTag;
import com.nlp.commons.ml.gm.linear.LinearSpanNetworkCompiler.nodeType;
import com.nlp.commons.types.InputToken;
import com.nlp.commons.types.Instance;
import com.nlp.commons.types.OutputToken;
import com.nlp.commons.types.Token;
import com.nlp.hybridnetworks.LocalNetworkParam;
import com.nlp.hybridnetworks.Network;
import com.nlp.hybridnetworks.NetworkIDMapper;



public class TargetSentimentCompiler extends LinearNetworkCompiler {

	public static int NEMaxLength = 6;
	int SpanMaxLength = 10;
	boolean visual = true;
	
	//TargetSentimentViewer viewer = new TargetSentimentViewer(this, null, 5);
	void visualize(LinearNetwork network, String title, int networkId)
	{
		//viewer.visualizeNetwork(network, null, title + "[" + networkId + "]");
	}
	
	public TargetSentimentCompiler() {
		super(null);
		// TODO Auto-generated constructor stub
	}
	
	public TargetSentimentCompiler(int NEMaxLength, int SpanMaxLength) {
		super(null);
		this.NEMaxLength = NEMaxLength;
		this.SpanMaxLength = SpanMaxLength;
		System.out.println("NEMaxLength=" + NEMaxLength);
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 2100499563741744475L;
	
	public enum NodeType {Start, Span, End};
	
	public enum SubNodeType {B, A}
	
	public enum PolarityType {positive, negative, neutral}
	
	private OutputToken[] _allOutputs;
	
	TargetSentimentFeatureManager fm;

	
	int PolarityTypeSize = PolarityType.values().length;
	
	int SubNodeTypeSize = SubNodeType.values().length;
	
	

	
	public void setOutputToken(OutputToken[] allOutputs)
	{
		this._allOutputs = allOutputs;
	}
	
	public void setFeatureManager(TargetSentimentFeatureManager fm)
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
			int polar = PolarityTypeSize - ids[1];
			int subnode = SubNodeTypeSize - ids[2];
			int node_type = ids[4];
			
			if(ids[4] == NodeType.Span.ordinal()){
				
				preds.add(new int[]{pos, polar, subnode});
				
			}
			
		}
		

		
		ArrayList<OutputTag> predication_array = new ArrayList<OutputTag>();
		PolarityType polar = null;
		int entity_begin = -1;
		
		for(int i = 0; i < preds.size(); i++)
		{
			int[] ids = preds.get(i);
			int pos = ids[0];
			int polar_index = ids[1];
			int subnode_index = ids[2];
			
			
			//System.out.println(pos + "," + SubNodeType.values()[subnode_index].name() 
			//		+ "," + PolarityType.values()[polar_index].name());
			
			
			//left node 
			if (subnode_index == SubNodeType.B.ordinal())
			{
				
				int[] next_ids =  preds.get(i + 1);
				int next_pos = next_ids[0];
				int next_polar_index = next_ids[1];
				int next_subnode_index = next_ids[2];
				
				//next node is before node
				if (next_subnode_index == SubNodeType.B.ordinal())
				{
					predication_array.add(new OutputTag("O"));
					
				} 
				else if (next_subnode_index == SubNodeType.A.ordinal())
				{
					//entity_begin  = pos;
					
					polar = PolarityType.values()[next_polar_index];
					
					predication_array.add(new OutputTag("B-" + polar.name()) );
					
					
					for(int k = pos + 1; k <= next_pos; k++)
					{
						predication_array.add(new OutputTag("I-" + polar.name()));	
					}
					
					
				}
					
				
				
			}
			else if (subnode_index == SubNodeType.A.ordinal())
			{
				if (pos < size - 1)
				{
					int[] next_ids =  preds.get(i + 1);
					int next_pos = next_ids[0];
					int next_polar_index = next_ids[1];
					int next_subnode_index = next_ids[2];
					
					//from After to next After
					if (next_subnode_index == SubNodeType.A.ordinal())
					{
						predication_array.add(new OutputTag("O"));
					}
				}
			}

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

		//OutputToken[] outputs = inst.getOutput();
		
		int size = inst.size();
		

		
		long start = this.toNode_start(size);
		network.addNode(start);
		
		long[][][] node_array = new long[size][PolarityTypeSize][SubNodeTypeSize];
		
		//build node array
		for(int pos = 0; pos < size; pos++)
		{
			for(int polar = 0; polar < PolarityTypeSize; polar++)
			{
				for(int sub = 0; sub < SubNodeTypeSize; sub++)
				{
					long node = this.toNode_Span(size, pos, polar, sub);
					network.addNode(node);
					node_array[pos][polar][sub] = node;
				}
			}
		}
		
		long end = this.toNode_end(inst.size());
		network.addNode(end);
		

		long from = -1, to = -1;
		
		
			
		//add first column of span node from start
		for(int j = 0; j < PolarityTypeSize; j++)
		{
			from = start;
			//System.out.println("inputs[0]:" + inputs[0].getName() + "\tj:" + j + "\tB:" + SubNodeType.B.ordinal());
			//System.out.println("\t" + node_array[0][j].length);
			to = node_array[0][j][SubNodeType.B.ordinal()];
			network.addEdge(from, new long[]{to});
		}
		
		

		
		for(int pos = 0; pos < size ; pos++)
		{
			for(int j = 0; j < PolarityTypeSize; j++)
			{
				//before to next before
				if (pos < size - 1)
				{
					from =  node_array[pos][j][SubNodeType.B.ordinal()];
					to = node_array[pos + 1][j][SubNodeType.B.ordinal()];
					network.addEdge(from, new long[]{to});
				}
				
				/*
				//before to current entity
				from =  node_array[pos][j][SubNodeType.B.ordinal()];
				to = node_array[pos][j][SubNodeType.e.ordinal()];
				network.addEdge(from, new long[]{to});
				
				
				
				
				//entity to after
				from =  node_array[pos][j][SubNodeType.e.ordinal()];
				to = node_array[pos][j][SubNodeType.A.ordinal()];
				network.addEdge(from, new long[]{to});*/
				
				//entity to next entity
				if (pos < size - 1)
				{
					for(int len = 0; len < NEMaxLength; len++)
						if (pos + len < size)
						{
							from =  node_array[pos][j][SubNodeType.B.ordinal()];
							to = node_array[pos + len][j][SubNodeType.A.ordinal()];
							network.addEdge(from, new long[]{to});
						}
				}
				
				
				
				//after to next after
				if (pos < size - 1)
				{
					from =  node_array[pos][j][SubNodeType.A.ordinal()];
					to = node_array[pos + 1][j][SubNodeType.A.ordinal()];
					network.addEdge(from, new long[]{to});
				}
				
				//after to next before
				if (pos < size - 1)
				{
				
					for(int k = 0; k < PolarityTypeSize; k++)
					{
						from =  node_array[pos][j][SubNodeType.A.ordinal()];
						to = node_array[pos + 1][k][SubNodeType.B.ordinal()];
						network.addEdge(from, new long[]{to});
						
					}
				}
				
				
				
			}
			
		}
		
		//add last column of span node to end
		for(int j = 0; j < PolarityTypeSize; j++)
		{
			from = node_array[size - 1][j][SubNodeType.A.ordinal()];
			to = end;
			network.addEdge(from, new long[]{to});
			
			from = node_array[size - 1][j][SubNodeType.B.ordinal()];
			to = node_array[size - 1][j][SubNodeType.A.ordinal()];
			network.addEdge(from, new long[]{to});
		}

		
	
		network.finalizeNetwork();
		
		if (visual)
			visualize(network, "Sentiment Model:unlabeled", networkId);
//		System.err.println(network.countNodes()+" nodes.");
//		System.exit(1);
		
		return network;
	}

	private LinearNetwork compileLabeled(int networkId, LinearInstance inst, LocalNetworkParam param){
		
		LinearNetwork network = new LinearNetwork(networkId, inst, param);
		
		InputToken[] inputs = inst.getInput();
		
		OutputToken[] outputs = inst.getOutput();
			
		int size = inst.size();
		
		int len = -1;
		
		long start = this.toNode_start(size);
		network.addNode(start);
		
		long[][][] node_array = new long[size][PolarityTypeSize][SubNodeTypeSize];
		
		//build node array
		for(int pos = 0; pos < size; pos++)
		{
			for(int polar = 0; polar < PolarityTypeSize; polar++)
			{
				for(int sub = 0; sub < SubNodeTypeSize; sub++)
				{
					long node = this.toNode_Span(size, pos, polar, sub);
					network.addNode(node);
					node_array[pos][polar][sub] = node;
				}
			}
		}
		
		long end = this.toNode_end(inst.size());
		network.addNode(end);
		
		
		////////////////////////////////////////////
		
		
		int last_entity_pos = -1;
		PolarityType last_polar = null;
		PolarityType polar = null;
		long from = -1;
		long to = -1;
		int entity_begin = -1;
		
		for(int pos = 0; pos < size; pos++)
		{
			String word = inputs[pos].getName();
			String label = outputs[pos].getName();
			
			boolean start_entity = this.startOfEntity(pos, size, outputs);
			boolean end_entity = this.endofEntity(pos, size, outputs);
			
		
			if (start_entity) {
				polar = PolarityType.valueOf(label.substring(2));
				
				/*
				from = node_array[pos][polar.ordinal()][SubNodeType.B.ordinal()];
				to = node_array[pos][polar.ordinal()][SubNodeType.e.ordinal()];
				network.addEdge(from,  new long[]{to});
				*/
				if (last_entity_pos == -1)
				{
					from = start;
					to = node_array[0][polar.ordinal()][SubNodeType.B.ordinal()];
					network.addEdge(from,  new long[]{to});
					
					/// directly from left to right
					for(int i = 0; i < pos; i++)
					{
						//from before node[pos] to before node at [pos+1]
						from = node_array[i][polar.ordinal()][SubNodeType.B.ordinal()];
						to = node_array[i + 1][polar.ordinal()][SubNodeType.B.ordinal()];
						network.addEdge(from,  new long[]{to});
						
					}
					
					
					
				} 
				else 
				{
					
					
								
					//latent path
					for(int i = last_entity_pos + 1; i < pos; i++)
					{
						//add A->A
						from = node_array[i - 1][last_polar.ordinal()][SubNodeType.A.ordinal()];
						to = node_array[i][last_polar.ordinal()][SubNodeType.A.ordinal()];
						network.addEdge(from,  new long[]{to});
						
						//add B->B
						from = node_array[i][polar.ordinal()][SubNodeType.B.ordinal()];
						to = node_array[i + 1][polar.ordinal()][SubNodeType.B.ordinal()];
						network.addEdge(from,  new long[]{to});
						
					}
					
					
					for(int i = last_entity_pos; i < pos; i++)
					{
						//add A->B
						from = node_array[i][last_polar.ordinal()][SubNodeType.A.ordinal()];
						to = node_array[i + 1][polar.ordinal()][SubNodeType.B.ordinal()];
						network.addEdge(from,  new long[]{to});
						
					}
					
					
					
				}
				
				
				entity_begin = pos;
				
			}
			
			if (end_entity) {
				
				//add links between entity
				/*
				for(int i = entity_begin; i < pos; i++)
				{
					from = node_array[i][polar.ordinal()][SubNodeType.e.ordinal()];
					to = node_array[i + 1][polar.ordinal()][SubNodeType.e.ordinal()];
					network.addEdge(from,  new long[]{to});
					
				}*/
				len = pos - entity_begin + 1;
				
				
				if (entity_begin <= pos)
				{
					if (len > this.NEMaxLength)
					{
						polar = null;
						break;
						/*int mid = (entity_begin + pos) / 2;
						from = node_array[entity_begin][polar.ordinal()][SubNodeType.B.ordinal()];
						to = node_array[mid][polar.ordinal()][SubNodeType.A.ordinal()];
						network.addEdge(from,  new long[]{to});
						
						from = node_array[mid][polar.ordinal()][SubNodeType.A.ordinal()];
						to = node_array[mid + 1][polar.ordinal()][SubNodeType.B.ordinal()];
						network.addEdge(from,  new long[]{to});
						
						from = node_array[mid + 1][polar.ordinal()][SubNodeType.B.ordinal()];
						to = node_array[pos][polar.ordinal()][SubNodeType.A.ordinal()];
						network.addEdge(from,  new long[]{to});
						*/
						
					}
					else
					{
						from = node_array[entity_begin][polar.ordinal()][SubNodeType.B.ordinal()];
						to = node_array[pos][polar.ordinal()][SubNodeType.A.ordinal()];
						network.addEdge(from,  new long[]{to});
					}
				}
				
				/*
				//add link from entity to After
				from = node_array[pos][polar.ordinal()][SubNodeType.e.ordinal()];
				to = node_array[pos][polar.ordinal()][SubNodeType.A.ordinal()];
				network.addEdge(from,  new long[]{to});
				*/
				
				last_entity_pos = pos;
				last_polar = polar;
				
			}
			
		}
		
		
		//add the last column node to end
		if (polar != null)
		{
			for(int pos = last_entity_pos + 1; pos < size; pos++)
			{
				from = node_array[pos - 1][polar.ordinal()][SubNodeType.A.ordinal()];
				to = node_array[pos][polar.ordinal()][SubNodeType.A.ordinal()];
				network.addEdge(from,  new long[]{to});
			}
			
			from = node_array[size - 1][polar.ordinal()][SubNodeType.A.ordinal()];
			to = end;
			network.addEdge(from,  new long[]{to});
		} else {
			//polar = PolarityType._;
			//from = start;
			//to = node_array[0][polar.ordinal()][SubNodeType.B.ordinal()];
			//network.addEdge(from,  new long[]{to});
			
			System.out.println("No Entity found or too long entity (len:" + len + ") in this Instance, Discard!");
			
			for(int pos = 0; pos < size; pos++)
			{
				System.out.print(inputs[pos].getName() + " ");
			}
			System.out.println();
			network = new LinearNetwork(networkId, inst, param);
			
			
			
			start = this.toNode_start(size);
			network.addNode(start);
			
			end = this.toNode_end(size);
			network.addNode(end);
			
			network.addEdge(start, new long[]{end});
			
		}
		
		
		
		network.finalizeNetwork();
		
		if (visual)
			visualize(network, "Sentiment Model:labeled", networkId);
		
//		System.err.println(network.countNodes()+" nodes.");
//		System.exit(1);
		
		return network;
		
	}
	
	boolean startOfEntity(int pos, int size, OutputToken[] outputs)
	{
		String label = outputs[pos].getName();
		if (label.startsWith("B"))
			return true;
		
		if (pos == 0 && label.startsWith("I"))
			return true;
		
		if (pos > 0)
		{
			String prev_label =  outputs[pos - 1].getName();
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
	
	private long toNode_Span(int size, int bIndex, int polar, int subnode){
		//System.out.println("bIndex=" + bIndex);
		return NetworkIDMapper.toHybridNodeID(new int[]{size - bIndex, PolarityTypeSize - polar, SubNodeTypeSize - subnode, 0, NodeType.Span.ordinal()});
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

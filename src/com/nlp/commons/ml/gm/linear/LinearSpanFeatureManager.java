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
package com.nlp.commons.ml.gm.linear;

import java.util.ArrayList;

import com.nlp.commons.ml.gm.linear.LinearSpanNetworkCompiler.nodeType;
import com.nlp.commons.types.AttWordToken;
import com.nlp.commons.types.Instance;
import com.nlp.commons.types.OutputToken;
import com.nlp.hybridnetworks.FeatureArray;
import com.nlp.hybridnetworks.FeatureManager;
import com.nlp.hybridnetworks.GlobalNetworkParam;
import com.nlp.hybridnetworks.Network;
import com.nlp.hybridnetworks.NetworkIDMapper;

public class LinearSpanFeatureManager extends FeatureManager{
	
	private static final long serialVersionUID = -7166073257274059610L;
	
	private enum FEATURE_TYPES {unigram, bigram, span, transition};
	
	protected OutputToken[] _outputTokens;
	
	public LinearSpanFeatureManager(GlobalNetworkParam param_g, OutputToken[] outputTokens) {
		super(param_g);
		this._outputTokens = outputTokens;
	}

	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k) {

		if(children_k.length>1)
			throw new RuntimeException("The number of children should be at most 1, but it is "+children_k.length);
		
		long node_parent = ((LinearSpanNetwork)network).getNode(parent_k);
		
		if(children_k.length == 0 )
			return FeatureArray.EMPTY;
		
		long node_child = ((LinearSpanNetwork)network).getNode(children_k[0]);
		
		int[] ids_parent = NetworkIDMapper.toHybridNodeArray(node_parent);
		int tagIndex_parent = ids_parent[3];
		
		int[] ids_child = NetworkIDMapper.toHybridNodeArray(node_child);
		int tagIndex_child = ids_child[3];
		
		OutputToken tag_prev = this._outputTokens[tagIndex_child];
		OutputToken tag_curr = this._outputTokens[tagIndex_parent];
		
		if(ids_parent[4]==nodeType.ROOT.ordinal()){
			return this.extract(network, tag_prev.getName(), tag_curr.getName());
		} else if(ids_parent[4]==nodeType.SPAN.ordinal()
				&& ids_child[4]==nodeType.END.ordinal()){
			int eIndex = ids_parent[0];
			int bIndex = ids_parent[1];
			return this.extract(network, bIndex, eIndex, tag_prev.getName(), tag_curr.getName());
		} else if(ids_parent[4]==nodeType.SPAN.ordinal()
				&& ids_child[4]==nodeType.LEAF.ordinal()){
			int eIndex = ids_parent[0];
			int bIndex = ids_parent[1];
			return this.extract(network, bIndex, eIndex, "*", tag_curr.getName());
		} else {
			return FeatureArray.EMPTY;
		}
		
	}
	
	//this is a very simple set of features.
	protected FeatureArray extract(Network network, int bIndex, int eIndex, String tag_prev_name, String tag_curr_name){
		Instance inst = network.getInstance();
		
		ArrayList<Integer> arr = new ArrayList<Integer>();
		
		int f;
		f = this._param_g.toFeature(FEATURE_TYPES.transition.name(),":"+tag_prev_name, tag_curr_name);
		arr.add(f);
		for(int wIndex = bIndex; wIndex<eIndex; wIndex++){
			AttWordToken word = (AttWordToken) ((LinearSpanInstance)inst).getInput()[wIndex];
			f = this._param_g.toFeature(FEATURE_TYPES.unigram.name(), tag_curr_name, word.getName());
			arr.add(f);
			if(wIndex==bIndex){
				f = this._param_g.toFeature(FEATURE_TYPES.bigram.name(), tag_curr_name, "|||"+word.getName());
				arr.add(f);
			} else {
				AttWordToken word1 = (AttWordToken) ((LinearSpanInstance)inst).getInput()[wIndex-1];
				f = this._param_g.toFeature(FEATURE_TYPES.bigram.name(), tag_curr_name, word1.getName()+"|||"+word.getName());
				arr.add(f);
			}
			if(wIndex==eIndex-1){
				f = this._param_g.toFeature(FEATURE_TYPES.bigram.name(), tag_curr_name, word.getName()+"|||");
				arr.add(f);
			}
		}
		
		StringBuilder sb = new StringBuilder();
		sb.append('[');
		for(int wIndex = bIndex; wIndex<eIndex; wIndex++){
			AttWordToken word = (AttWordToken) ((LinearSpanInstance)inst).getInput()[wIndex];
			sb.append('|');
			sb.append(word.getName());
		}
		sb.append(']');
		f = this._param_g.toFeature(FEATURE_TYPES.span.name(), tag_curr_name, sb.toString());
		arr.add(f);
		
		int fs[] = new int[arr.size()];
		for(int k = 0; k<arr.size(); k++)
			fs[k] = arr.get(k);
		
		return new FeatureArray(fs);
	}
	
	//this is a very simple set of features.
	protected FeatureArray extract(Network network, String tag_prev_name, String tag_curr_name){
		int f = this._param_g.toFeature(FEATURE_TYPES.transition.name(), tag_prev_name, tag_curr_name);
		return new FeatureArray(new int[]{f});
	}
	
}
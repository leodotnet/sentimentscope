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

import com.nlp.commons.ml.gm.linear.LinearNetworkCompiler.nodeType;
import com.nlp.commons.types.AttWordToken;
import com.nlp.commons.types.Instance;
import com.nlp.commons.types.OutputToken;
import com.nlp.hybridnetworks.FeatureArray;
import com.nlp.hybridnetworks.FeatureManager;
import com.nlp.hybridnetworks.Network;
import com.nlp.hybridnetworks.NetworkIDMapper;
import com.nlp.hybridnetworks.GlobalNetworkParam;

public class LinearFeatureManager extends FeatureManager{
	
	private static final long serialVersionUID = -7336453398566775362L;
	
	private enum FEATURE_TYPES {emission, transition};
	
	protected OutputToken[] _outputTokens;
	
	public LinearFeatureManager(GlobalNetworkParam param, OutputToken[] outputTokens) {
		super(param);
		this._outputTokens = outputTokens;
	}
	
	@Override
	public FeatureArray extract_helper(Network network, int parent_k, int[] children_k) {
		
		if(children_k.length>1)
			throw new RuntimeException("The number of children should be at most 1, but it is "+children_k.length);
		
		long node_parent = ((LinearNetwork)network).getNode(parent_k);
		
		if(children_k.length == 0 )
			return FeatureArray.EMPTY;
		
		long node_child = ((LinearNetwork)network).getNode(children_k[0]);
		
		int[] ids_parent = NetworkIDMapper.toHybridNodeArray(node_parent);
		int tagIndex_parent = ids_parent[3];
		
		int[] ids_child = NetworkIDMapper.toHybridNodeArray(node_child);
		int tagIndex_child = ids_child[3];
		
		String tagname_prev;
		if(ids_child[4]==nodeType.LEAF.ordinal()){
			tagname_prev = "<START>";
		} else {
			tagname_prev = this._outputTokens[tagIndex_child].getName();
		}
		
		String tagname_curr = this._outputTokens[tagIndex_parent].getName();
		if(ids_parent[4]==nodeType.ROOT.ordinal()){
			return this.extract(network, tagname_prev, "<FINISH>");
		}
		
		int wordIndex = ids_parent[0]-1;
		return this.extract(network, wordIndex, tagname_prev, tagname_curr);
	}
	
	//this is a very simple set of features.
	protected FeatureArray extract(Network network, int wordIndex, String tagname_prev, String tagname_curr){
		Instance inst = network.getInstance();
		AttWordToken word = (AttWordToken) ((LinearInstance)inst).getInput()[wordIndex];
		int f1 = this._param_g.toFeature(FEATURE_TYPES.emission.name(), tagname_curr, word.getName());
		int f2 = this._param_g.toFeature(FEATURE_TYPES.transition.name(), tagname_prev, tagname_curr);
		return new FeatureArray(new int[]{f1, f2});
	}
	
	//this is a very simple set of features.
	protected FeatureArray extract(Network network, String tagname_prev, String tagname_curr){
		int f = this._param_g.toFeature(FEATURE_TYPES.transition.name(), tagname_prev, tagname_curr);
		return new FeatureArray(new int[]{f});
	}
	
}
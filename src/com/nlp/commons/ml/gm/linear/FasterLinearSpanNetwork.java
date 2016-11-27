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

import com.nlp.hybridnetworks.LocalNetworkParam;

public class FasterLinearSpanNetwork extends LinearSpanNetwork{
	
	private static final long serialVersionUID = -1207303047086257934L;
	private int _num_nodes;

	public FasterLinearSpanNetwork(int networkId, LinearSpanInstance inst, LocalNetworkParam param) {
		super(networkId, inst, param);
		throw new RuntimeException("invalid");
	}
	
	public FasterLinearSpanNetwork(int networkId, LinearSpanInstance inst, long[] nodes, int[][][] children, LocalNetworkParam param, int num_nodes) {
		super(networkId, inst, nodes, children, param);
		this._num_nodes = num_nodes;
	}
	
	@Override
	public int countNodes(){
		return this._num_nodes;
	}

	//remove the node k from the network.
	@Override
	public void remove(int k){
		//DO NOTHING..
	}
	
	//check if the node k is removed from the network.
	@Override
	public boolean isRemoved(int k){
		return false;
	}
	
}
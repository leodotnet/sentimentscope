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
/**
 * 
 */
package com.nlp.hybridnetworks;

import java.io.Serializable;

/**
 * @author wei_lu
 *
 */
public interface HyperGraph extends Serializable{
	
	//count the total number of nodes.
	public int countNodes();
	
	//get the node with index k.
	public long getNode(int k);
	
	public int[] getNodeArray(int k);
	
	//get the children for node with index k.
	public int[][] getChildren(int k);
	
	//check whether the node with index k is removed.
	public boolean isRemoved(int k);
	
	//remove the node with index k.
	public void remove(int k);
	
	//check whether the node with index k is the root of the network.
	public boolean isRoot(int k);
	
	//check whether the node with index k is a leaf of the network.
	public boolean isLeaf(int k);
	
	//check if the network contains a particular node.
	public boolean contains(long node);
	
}

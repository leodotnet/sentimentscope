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
package com.nlp.commons;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

//tree is a special kind of forest.
public class Forest implements Serializable{
	
	private static final long serialVersionUID = -1094223168650826764L;
	
	protected HashMap<TreeNode, ArrayList<TreeNode[]>> _map;
	protected TreeNode _root;
	
	public Forest(){
		this._map = new HashMap<TreeNode, ArrayList<TreeNode[]>>();
	}
	
	public ArrayList<TreeNode> getAllTreeNodes(){
		ArrayList<TreeNode> nodes = new ArrayList<TreeNode>();
		Iterator<TreeNode> keys = this._map.keySet().iterator();
		while(keys.hasNext())
			nodes.add(keys.next());
		return nodes;
	}
	
	//warning: if there is a loop, this method will never terminate.
	//avoid using it if it is not a tree.
	public int countAllChildren(TreeNode node){
		if(this.getChildren(node)==null){
			return -1;
		}
		int count = 0;
		ArrayList<TreeNode[]> childrenList = this.getChildren(node);
		for(TreeNode[] children : childrenList){
			count += children.length;
			for(TreeNode child : children){
				count += this.countAllChildren(child);
			}
		}
		return count;
	}
	
	//warning: this method does not check whether the nodes are connected.
	//so, theoretically, it is checking whether this forest is a
	//tree fragment or not.
	public boolean isTree(){
		Iterator<TreeNode> nodes = this._map.keySet().iterator();
		while(nodes.hasNext()){
			TreeNode node = nodes.next();
			if(this._map.get(node).size()>1){
				System.err.println(node.toString()+"<<<<<<"+this._map.get(node).size()+"\t"+this._map.get(node).get(0).length);
				return false;
			}
		}
		return true;
	}
	
	//get the root node of the forest.
	public TreeNode getRoot(){
		return this._root;
	}
	
	//count the number of nodes in the forest.
	public int countNodes(){
		return this._map.size();
	}
	
	public void addNode(TreeNode node, boolean isRoot){
		if(isRoot) this._root = node;
		if(this._map.containsKey(node))
			return;
		this._map.put(node, new ArrayList<TreeNode[]>());
	}
	
	public void addLink(TreeNode parent, TreeNode[] children){
		if(!this._map.containsKey(parent))
			this._map.put(parent, new ArrayList<TreeNode[]>());
		this._map.get(parent).add(children);
	}
	
	public ArrayList<TreeNode[]> getChildren(TreeNode parent){
		return this._map.get(parent);
	}
	
	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		
		Iterator<TreeNode> nodes = this._map.keySet().iterator();
		while(nodes.hasNext()){
			TreeNode node = nodes.next();
			sb.append(node);
			sb.append('\n');
			ArrayList<TreeNode[]> childrenList = this._map.get(node);
			for(TreeNode[] children : childrenList){
				sb.append('\t');
				sb.append('[');
				for(int k = 0; k<children.length; k++){
					if(k!=0)
						sb.append(',');
					sb.append(children[k]);
				}
				sb.append(']');
				sb.append('\n');
			}
			sb.append('\n');
		}
		
		return sb.toString();
	}
	
}

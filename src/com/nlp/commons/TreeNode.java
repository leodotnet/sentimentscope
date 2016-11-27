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

import com.nlp.commons.types.Token;

//this is a node in the hybrid network.

public abstract class TreeNode implements Token{
	
	private static final long serialVersionUID = 2045193211729928725L;
	
	protected String _form;
	
	public TreeNode(String form){
		this._form = form;
	}
	
	public abstract int arity();
	
	@Override
	public String getName(){
		return this._form;
	}
	
	@Override
	public boolean equals(Object o){
		if(o instanceof TreeNode)
			return this._form.equals(((TreeNode)o)._form);
		return false;
	}
	
	@Override
	public int hashCode(){
		return this._form.hashCode();
	}
	
	//print a one-line information for this node.
	@Override
	public String toString(){
		return this._form;
	}
	
}
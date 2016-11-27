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

import java.util.Arrays;

import com.nlp.commons.types.Token;

public class Attribute implements Token{
	
	private static final long serialVersionUID = -2704233161552245781L;
	
	private String[] _forms;
	
	public Attribute(String[] forms){
		this._forms = forms;
	}
	
	@Override
	public int getId() {
		return 0;
	}
	
	@Override
	public int hashCode(){
		return Arrays.hashCode(this._forms);
	}
	
	@Override
	public boolean equals(Object o){
		if(o instanceof Attribute){
			Attribute at = (Attribute)o;
			return Arrays.equals(this._forms, at._forms);
		}
		return false;
	}
	
	@Override
	public String getName() {
		return Arrays.toString(this._forms);
	}
	
	@Override
	public String toString(){
		return Arrays.toString(this._forms);
	}
	
}

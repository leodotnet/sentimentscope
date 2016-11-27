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
package com.nlp.hybridnetworks;

import java.io.Serializable;

public class Pattern implements Serializable{
	
	private static final long serialVersionUID = -4840102291209401455L;
	
	private String _form;
	
	public Pattern(String form){
		this._form = form;
		this._minWords = 0;
		char ch[] = this._form.toCharArray();
		for(char c : ch){
			if(c=='w' || c=='W'){
				this._minWords ++;
			}
		}
		this._length = this._form.length();
	}
	
	private int _minWords;
	private int _length;
	
	public int countMinWords(){
		return _minWords;
	}
	
	public String getForm(){
		return this._form;
	}
	
	public int length(){
		return this._length;
	}
	
	public boolean isW(){
		return this._form.equals("W");
	}
	
	public boolean isw(){
		return this._form.equals("w");
	}
	
	public boolean isZ(){
		return this._form.equals("Z");
	}
	
	public boolean isX(){
		return this._form.equals("X");
	}
	
	public boolean isY(){
		return this._form.equals("Y");
	}
	
	public boolean isEmpty(){
		return this._form.equals("-");
	}
	
	@Override
	public boolean equals(Object o){
		if(o instanceof Pattern)
			return this._form.equals(((Pattern)o)._form);
		return false;
	}
	
	@Override
	public int hashCode(){
		return this._form.hashCode() + 7;
	}
	
	@Override
	public String toString(){
		return this._form;
	}
	
}
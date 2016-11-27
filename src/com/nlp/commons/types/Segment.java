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
package com.nlp.commons.types;

import java.io.Serializable;

public class Segment implements Serializable, Comparable<Segment>{
	
	private static final long serialVersionUID = 7669804479975787795L;
	private int _bIndex;
	private int _eIndex;
	
	public Segment(int bIndex, int eIndex){
		this._bIndex = bIndex;
		this._eIndex = eIndex;
	}
	
	public int getBIndex(){
		return this._bIndex;
	}
	
	public int getEIndex(){
		return this._eIndex;
	}
	
	public int length(){
		return this._eIndex - this._bIndex;
	}
	
	public boolean nestedWith(Segment seg){
		if(this._bIndex >= seg._bIndex && this._eIndex <= seg._eIndex){
			return true;
		}
		if(this._bIndex <= seg._bIndex && this._eIndex >= seg._eIndex){
			return true;
		}
		return false;
	}
	
	public boolean overlapsWith(Segment seg){
		if(this._bIndex<=seg._bIndex && this._eIndex>seg._bIndex){
			return true;
		}
		if(this._bIndex<seg._eIndex && this._eIndex>=seg._eIndex){
			return true;
		}
		if(seg._bIndex<=this._bIndex && seg._eIndex>this._bIndex){
			return true;
		}
		if(seg._bIndex<this._eIndex && seg._eIndex>=this._eIndex){
			return true;
		}
		return false;
	}
	
	public boolean noOverlapWith(Segment seg){
		if(this._eIndex<=seg._bIndex){
			return true;
		}
		if(seg._eIndex<=this._bIndex){
			return true;
		}
		return false;
	}
	
	@Override
	public int compareTo(Segment seg) {
		if(this._bIndex!=seg._bIndex)
			return this._bIndex - seg._bIndex;
		if(this._eIndex!=seg._eIndex)
			return this._eIndex - seg._eIndex;
		return 0;
	}
	
	@Override
	public int hashCode(){
		return (this._bIndex + 7) ^ (this._eIndex + 7);
	}
	
	@Override
	public boolean equals(Object o){
		if(o instanceof Segment){
			Segment seg = (Segment)o;
			return this._bIndex == seg._bIndex && this._eIndex == seg._eIndex;
		}
		return false;
	}
	
	@Override
	public String toString(){
		return "["+this._bIndex+","+this._eIndex+")";
	}

}
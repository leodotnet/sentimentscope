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
import java.util.Arrays;

public class HybridPattern implements Serializable{
	
	private static final long serialVersionUID = 957963248917237961L;
	
	//src pattern, tgt pattern
	protected int _id = -1;
	protected Pattern[] _patterns;
	
	protected HybridPattern(Pattern[] patterns){
		assert patterns.length == 2;
		this._patterns = patterns;
	}
	
	public Pattern getSrcPattern(){
		return this._patterns[0];
	}
	
	public Pattern getTgtPattern(){
		return this._patterns[1];
	}
	
	public void setId(int id){
		this._id = id;
	}
	
	public int getId(){
		return this._id;
	}
	
	public boolean isZZ(){
		return this.getSrcPattern().isZ() && this.getTgtPattern().isZ();
	}
	
	public boolean isXX(){
		return this.getSrcPattern().isX() && this.getTgtPattern().isX();
	}
	
	public boolean isYY(){
		return this.getSrcPattern().isY() && this.getTgtPattern().isY();
	}
	
	public boolean isXY(){
		return this.getSrcPattern().isX() && this.getTgtPattern().isY();
	}
	
	public boolean isYX(){
		return this.getSrcPattern().isY() && this.getTgtPattern().isX();
	}
	
	@Override
	public boolean equals(Object o){
		if(o instanceof HybridPattern)
			return Arrays.equals(this._patterns, ((HybridPattern)o)._patterns);
		return false;
	}
	
	@Override
	public int hashCode(){
		return Arrays.hashCode(this._patterns) + 7;
	}
	
	@Override
	public String toString(){
		return Arrays.toString(this._patterns);
	}
	
}
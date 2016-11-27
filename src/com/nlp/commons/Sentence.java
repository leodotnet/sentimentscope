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

public class Sentence implements Serializable{
	
	private static final long serialVersionUID = 3412475193204872902L;
	
	protected String _language;
	protected Word[] _words;
	
	public Sentence(String language, Word[] words){
		this._language = language;
		this._words = words;
	}
	
	public String getLanguage(){
		return this._language;
	}
	
	public Word[] getWords(){
		return this._words;
	}
	
	public int length(){
		return this._words.length;
	}
	
	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append("LANG:");
		sb.append(this._language);
		sb.append('[');
		for(int k = 0; k<this._words.length; k++){
			if(k!=0)
				sb.append(',');
			sb.append(this._words[k]);
		}
		sb.append(']');
		return sb.toString();
	}
	
}
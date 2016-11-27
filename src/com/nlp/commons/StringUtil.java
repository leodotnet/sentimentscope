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

import java.util.StringTokenizer;

public class StringUtil {
	
	public static void main(String args[]){
		String input = "<sentence><cons lex=\"adenovirus_(Ad)_infection\" sem=\"G#other_name\"><cons lex=\"adenovirus\" sem=\"G#virus\">Adenovirus</cons> (Ad)-infection</cons>and <cons lex=\"E1A_transfection\" sem=\"G#other_name\"><cons lex=\"E1A\" sem=\"G#protein_molecule\">E1A</cons> transfection</cons> were used to model changes in susceptibility to <cons lex=\"NK_cell_killing\" sem=\"G#other_name\">NK cell killing</cons> caused by transient vs stable <cons lex=\"E1A_expression\" sem=\"G#other_name\"><cons lex=\"E1A\" sem=\"G#protein_molecule\">E1A</cons> expression</cons> in <cons lex=\"human_cell\" sem=\"G#cell_type\">human cells</cons>.</sentence>";
		String output = stripXMLTags(input);
		System.err.println(output);
	}
	
	public static int numTokens(String input){
		StringTokenizer st = new StringTokenizer(input);
		return st.countTokens();
	}
	
	public static String stripSpaces(String input){
		StringTokenizer st = new StringTokenizer(input);
		StringBuilder sb = new StringBuilder();
		while(st.hasMoreTokens()){
			sb.append(" ");
			sb.append(st.nextToken());
		}
		
		return sb.toString().trim();
	}
	
	public static String stripXMLTags(String input){
		StringBuilder sb = new StringBuilder();
		
		boolean record = true;
		char[] chs = input.toCharArray();
		for(char ch : chs){
			if(ch=='<'){
				sb.append(' ');
				record = false;
			} else if(ch=='>'){
				sb.append(' ');
				record = true;
			} else if(record){
				if(ch=='-' || ch=='/'){
					sb.append(' ');
					sb.append(ch);
					sb.append(' ');
				} else {
					sb.append(ch);
				}
			}
		}
		
		String output = sb.toString();
		
		return stripSpaces(output);
	}
	
}

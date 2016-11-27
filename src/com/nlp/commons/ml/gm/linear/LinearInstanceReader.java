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

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

import com.nlp.commons.types.AttWordToken;
import com.nlp.commons.types.InputToken;
import com.nlp.commons.types.OutputToken;

public class LinearInstanceReader {
	
	private static HashMap<OutputTag, OutputTag> tagsMap = new HashMap<OutputTag, OutputTag>();
	private static ArrayList<OutputTag> tagsList = new ArrayList<OutputTag>();
	
	public static LinearInstance[] readTest_column(String data_filename) throws IOException{
		LinearInstance[] insts = readLabeled_column(data_filename);
		for(LinearInstance inst : insts)
			inst.setUnlabeled();
		return insts;
	}
	
	public static LinearInstance[] readTrain_column(String data_filename) throws IOException{
		LinearInstance[] insts = readLabeled_column(data_filename);
		for(LinearInstance inst : insts)
			inst.setLabeled();
		return insts;
	}
	
	//read the column format used by crf++
	private static LinearInstance[] readLabeled_column(String data_filename) throws IOException{
		
		InputStreamReader in = new InputStreamReader(new FileInputStream(data_filename), "UTF-8");
		BufferedReader scan = new BufferedReader(in);
		
		ArrayList<LinearInstance> instances = new ArrayList<LinearInstance>();
		
		int id = 1;
		String line;
		while((line=scan.readLine())!=null){
			ArrayList<String> lines = new ArrayList<String>();
			while(true){
				if(line==null)
					break;
				if(line.trim().equals("")){
					line = scan.readLine();
					continue;
				}
				lines.add(line.trim());
				line = scan.readLine();
				if(line!=null){
					line = line.trim();
				}
				if(line==null || line.equals("")){
					InputToken inputs[] = new InputToken[lines.size()];
					OutputToken outputs[] = new OutputToken[lines.size()];
					for(int k = 0; k<lines.size(); k++){
						String[] tokens = lines.get(k).split("\\s");
						inputs[k] = toAttWord(tokens, tokens.length-1);
						outputs[k] = toOutputTag(tokens[tokens.length-1]);
					}
					LinearInstance inst = new LinearInstance(id++, 1.0, inputs, outputs);
					instances.add(inst);
					
					line = scan.readLine();
					if(line==null)
						break;
					
					lines = new ArrayList<String>();
				}
			}
		}
		scan.close();
		
		LinearInstance[] instances_arr = new LinearInstance[instances.size()];
		for(int k = 0; k<instances_arr.length; k++){
			instances_arr[k] = instances.get(k);
		}
		
		return instances_arr;
		
	}
	
	public static OutputTag[] getOutputTags(){
		OutputTag[] tags = new OutputTag[tagsList.size()];
		for(int k = 0; k<tags.length; k++){
			tags[k] = tagsList.get(k);
		}
		return tags;
	}
	
	public static OutputTag toOutputTag(String form){
		OutputTag tag = new OutputTag(form);
		if(tagsMap.containsKey(tag)){
			return tagsMap.get(tag);
		} else {
			tag.setId(tagsMap.size());
			tagsMap.put(tag, tag);
			tagsList.add(tag);
			return tag;
		}
	}
	
	private static AttWordToken toAttWord(String[] tokens, int endIndex){
		String name = tokens[0];
		AttWordToken word = new AttWordToken(name);
		word.addAtt("word", name);
		for(int k = 1; k<endIndex; k++){
			String token = tokens[k];
			int index = token.indexOf("@");
			String attName, attValue;
			if(index >=0){
				attName = token.substring(0, index).trim();
				attValue = token.substring(index+1).trim();
			} else {
				attName = "att-"+k;
				attValue = token;
			}
			word.addAtt(attName, attValue);
		}
		return word;
	}
	
}

package com.nlp.global;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Scanner;

import com.nlp.targetedsentiment.util.TargetSentimentGlobal;

public class BrownCluster {
	
	public static BrownCluster brown_cluster;
	
	HashMap<String, String> brown_cluster_map = new HashMap<String, String>();

	public BrownCluster(String lang) {
		if (lang.equals("en"))
			getWord2VecEnglish();
		else if (lang.equals("es"))
			getWord2VecSpanish();
		else
		{
			System.err.println("Language " + lang + " not found!");
		}
	}
	
	public HashMap<String, String> getWord2VecEnglish() {
		return getWord2Vec("data//Twitter_en//feature_files//brown_clusters");
	}

	public HashMap<String, String> getWord2VecSpanish() {
		return getWord2Vec("data//Twitter_es//feature_files//brown_clusters");
	}
	
	public HashMap<String, String> getWord2Vec(String filename) {
		File f = new File(filename);
		Scanner scanner = null;
		String word, brown_cluster_code ,line;
		
		String[] items = null;
		
		
		try {
			scanner = new Scanner(f);
			
			while (scanner.hasNextLine()) {
				line = scanner.nextLine();
			
				if (line.trim().length() == 0)
					continue;
				items = line.split("\t");
				
				word = items[1];
				brown_cluster_code = items[0];
				brown_cluster_map.put(word, brown_cluster_code);
				
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 

		scanner.close();
		
		

		return brown_cluster_map;

	}
	
	public String getBrownClusterCode(String word)
	{
		return this.brown_cluster_map.get(word);
	}

}

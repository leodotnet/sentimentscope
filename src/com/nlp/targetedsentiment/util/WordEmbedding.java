package com.nlp.targetedsentiment.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.Scanner;

import com.nlp.algomodel.AlgoModel;
import com.nlp.commons.types.InputToken;
import com.nlp.commons.types.Instance;
import com.nlp.hybridnetworks.NetworkConfig;



public class WordEmbedding {
	
	public static String WORD_EMBEDDING_FEATURE = "<WORD_EMBEDDING>";
	public static WordEmbedding Word2Vec;

	public int Size = -1;
	public int ShapeSize = -1;
	HashMap<String, Double[]> word2vec = null;
	String[] Words;
	HashMap<String, Integer> word2ID;
	public AlgoModel algomodel = null;
	public String[] labels;
	public static int word_embedding_feature_size = -1;
	
	public ArrayList<String> unknown_words = new ArrayList<String>();
	public ArrayList<String> known_words = new ArrayList<String>();
	
	//public static double[] initWeight;
	public static int NUM_INIT_WEIGHT = 10;
	
	public static void main(String[] args) throws FileNotFoundException
	{
		WordEmbedding word2vec = new WordEmbedding("en");
		
		/*
		Scanner scanner = new Scanner(System.in);
		while(scanner.hasNext())
		{
	       String line = scanner.nextLine();
	       String word = line.trim();
	       if (word.equals("exit"))
	    	   break;

	       Double[] vector = word2vec.getVector(word);
	       Integer ID = word2vec.getWordID(word);
	       
	       if (vector != null)
	       System.out.println(word2vec.getWord(ID) + "(" + ID + ")\t" + Arrays.toString(vector));
	       else
	    	   System.out.println("not exist");
	       
		}

		scanner.close();     */
		
		Scanner scanner = new Scanner(new File("data//Twitter_en//train.1.coll"));
		PrintWriter p = new PrintWriter(new File("data//Twitter_en//train.11.coll"));
		while(scanner.hasNextLine())
		{
			String line = scanner.nextLine();
			if (line.trim().length() == 0)
			{
				p.write("\n");
				continue;
			}
			String[] fields = line.split("\t");
			
			String word = fields[0];
			
			Double[] v = word2vec.getVector(word);
			
			if (v != null)
			{
				p.write(line + "\n");
			}
			else
			{
				String out = "";
				fields[0] = "<UNK>";
				out = fields[0];
				for(int i = 1; i < fields.length; i++)
				{
					out += "\t" + fields[i];
				}
				
				p.write(out + "\n");
			}
			
		}
		
		p.close();
		scanner.close();
		
	      
	}
	
	public WordEmbedding(String lang)
	{
		System.out.println("Loading Word Embedding for Language: " + lang);
		if (lang.equals("en"))
			getWord2VecEnglish();
		else
			getWord2VecSpanish();
		System.out.println("Word Embedding Loaded");
		
			
	}
	
	public Double[] getVector(String word)
	{
		if (word2vec != null)
			return word2vec.get(word);
		
		return null;
	}

	public HashMap<String, Double[]> getWord2VecEnglish() {
		return getWord2Vec("models//polyplot-en.dict");
	}

	public HashMap<String, Double[]> getWord2VecSpanish() {
		return getWord2Vec("models//polyplot-es.dict");
	}

	public HashMap<String, Double[]> getWord2Vec(String filename) {
		File f = new File(filename);
		Scanner scanner = null;
		String word, line;
		int pWords = 0;
		String[] items = null;
		word2vec = new HashMap<String, Double[]>();
		word2ID = new HashMap<String, Integer>();
		try {
			scanner = new Scanner(f);
			
			line = scanner.nextLine();
			items = line.split("\t");
			
			Size = Integer.parseInt(items[0]);
			Words = new String[Size];
			ShapeSize =Integer.parseInt(items[1]);
			
			/*
			if (TargetSentimentGlobal.DEFEAULT_SHAPE_SIZE != -1)
				ShapeSize = TargetSentimentGlobal.DEFEAULT_SHAPE_SIZE;
			*/
			System.out.println("Size=" + Size + "\tShapeSize=" + ShapeSize);
			
			while (scanner.hasNextLine()) {
				line = scanner.nextLine();
			
				if (line.trim().length() == 0)
					continue;
				items = line.split("\t");
				/*
				if (items.length != ShapeSize + 1)
				{
					System.out.println("Discard " + items[0]);
					continue;
				}*/
				word = items[0];
				
				
				Words[pWords] = word;
				word2ID.put(word, pWords);
				
				Double[] vector = new Double[ShapeSize];
				for (int i = 0; i < vector.length; i++) {
					vector[i] = Double.parseDouble(items[i + 1]);
				}

				word2vec.put(word, vector);
			
				pWords++;
				
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 

		scanner.close();
		
		

		return word2vec;

	}
	
	public Integer getWordID(String word)
	{
		return word2ID.get(word);
	}
	
	public String getWord(int id)
	{
		return this.Words[id];
	}
	
	
	public void setAlgomodel(AlgoModel algomodel)
	{
		this.algomodel = algomodel;
	}
	
	public void createWordEmbeddingFeature()
	{
		
		if (algomodel != null)
		{
			for(int i = 0; i < this.ShapeSize; i++)
			{
				for(String Out : labels)
				{
					int f_index = algomodel.param.toFeature(TargetSentimentGlobal.Word2Vec.WORD_EMBEDDING_FEATURE, Out, "" + i);		
					//System.out.println(word + "-" + Out + ":" + f_index);
					
				}
			}
		}
		
		word_embedding_feature_size = ShapeSize * labels.length;
		
	}
	
	public void setNumOutput(int numOutput)
	{
		this.labels = new String[numOutput];
	}

}

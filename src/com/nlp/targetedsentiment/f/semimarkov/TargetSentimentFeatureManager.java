package com.nlp.targetedsentiment.f.semimarkov;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;













import com.nlp.commons.ml.gm.linear.LinearFeatureManager;
import com.nlp.commons.ml.gm.linear.LinearNetwork;


import com.nlp.commons.types.InputToken;
import com.nlp.commons.types.Instance;
import com.nlp.commons.types.OutputToken;
import com.nlp.hybridnetworks.FeatureArray;
import com.nlp.hybridnetworks.GlobalNetworkParam;
import com.nlp.hybridnetworks.Network;
import com.nlp.hybridnetworks.NetworkIDMapper;
import com.nlp.targetedsentiment.f.semimarkov.TargetSentimentCompiler.*;
import com.nlp.targetedsentiment.util.TargetSentimentGlobal;
import com.nlp.targetedsentiment.util.WordWithFeatureToken;




public class TargetSentimentFeatureManager extends LinearFeatureManager {
	

	public TargetSentimentFeatureManager(GlobalNetworkParam param_g) {
		super(param_g, null);
	}
	

	/**
	 * 
	 */
	private static final long serialVersionUID = 1376229168119316535L;
	
	public enum FEATURE_TYPES {Unigram, Bigram, Trigram, Fourgram, Transition}
	
	public enum WORD_FEATURE_TYPES {
		_id_, ne, brown_clusters5, jerboa, brown_clusters3, _sent_, _sent_ther_, sentiment, postag
	};
	
	int PolarityTypeSize = PolarityType.values().length;
	int SubNodeTypeSize = SubNodeType.values().length;
	int WordFeatureTypeSize = WORD_FEATURE_TYPES.values().length;
	
	
	static HashMap<String, ArrayList<String>> LinguisticFeaturesLibaryList = new HashMap<String, ArrayList<String>>();
	
	protected InputToken[] _inputTokens;
	protected OutputToken[] _outputTokens;
	
	public void setInputTokens(InputToken[] inputTokens)
	{
		this._inputTokens = inputTokens;
	}
	
	public void setOutputTokens(OutputToken[] outputTokens)
	{
		this._outputTokens = outputTokens;
	}
	
	
	public void loadLinguisticFeatureLibary()
	{
		LinguisticFeaturesLibaryList.clear();
		
		Scanner scan = null;
		for(String filename : TargetSentimentGlobal.LinguisticFeaturesLibaryName)
		{
			ArrayList<String> libarylist = new ArrayList<String>();
			try {
				scan = new Scanner(new File(TargetSentimentGlobal.feature_file_path + filename));
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			
			while(scan.hasNextLine())
			{
				libarylist.add(scan.nextLine().trim());
			}
			
			scan.close();
			
			LinguisticFeaturesLibaryList.put(filename, libarylist);
		}
	}
	
	@Override
	public FeatureArray extract_helper(Network network, int parent_k, int[] children_k) {
		
		if(children_k.length>1)
			throw new RuntimeException("The number of children should be at most 1, but it is "+children_k.length);
		
		long node_parent = ((LinearNetwork)network).getNode(parent_k);
		
		if(children_k.length == 0)
			return FeatureArray.EMPTY;
		
		Instance inst = network.getInstance();
		
		this.setInputTokens((InputToken[])inst.getInput());
		this.setOutputTokens((OutputToken[])inst.getOutput());
		
		int size = inst.size();
		
		long node_child = ((LinearNetwork)network).getNode(children_k[0]);
		
		int[] ids_parent = NetworkIDMapper.toHybridNodeArray(node_parent);
		int[] ids_child = NetworkIDMapper.toHybridNodeArray(node_child);

		
		WordWithFeatureToken[] input_token = (WordWithFeatureToken[]) inst.getInput();
		
		int pos_parent = size - ids_parent[0];
		int pos_child = size - ids_child[0];
		
		int polar_parent = PolarityTypeSize - ids_parent[1];
		int polar_child = PolarityTypeSize - ids_child[1];
		
		int subnode_parent = SubNodeTypeSize - ids_parent[2];
		int subnode_child = SubNodeTypeSize - ids_child[2];
		
		int nodetype_parent = ids_parent[4];
		int nodetype_child = ids_child[4];
		
		int num_sent = 0;

		String word_parent = "";
		String word_child = "";
		
		if (pos_parent >= 0 && pos_parent < size)
			word_parent = input_token[pos_parent].getName();
		
		if (pos_child >= 0 && pos_child < size)
			word_child = input_token[pos_child].getName();
		

	
		FeatureArray fa = FeatureArray.EMPTY;
		
		ArrayList<Integer> feature = new ArrayList<Integer>();
		ArrayList<String[]> featureArr = new ArrayList<String[]>();
		
		String Out = "O";
		String prefix = "";
		
		LinearNetwork mynetwork = (LinearNetwork) network;
		if (mynetwork.feature == null) {
			this.get_features(mynetwork, input_token, size);
			
		}
		
		
		
		
		String prev_prev_feature="", prev_feature ="", current_feature ="", next_feature = "";
		String prev_brown_cluster = "<START>", current_brown_cluster ="<NULL>", next_brown_cluster="<END>";
		String prev_jerboa = "<START>", current_jerboa = "<NULL>", next_jerboa = "<END>";
		String[] previous_polarity = new String[]{"<START>", "<NULL>", "<END>"};
		String[] ther_sentiment_polarity = new String[]{"<START>", "<NULL>", "<END>"};
		
		current_feature = this.clean(word_parent);
		
		if (pos_parent >= 0 && pos_parent < size)
		{
			current_brown_cluster = input_token[pos_parent].feature.get(4);
			current_jerboa = input_token[pos_parent].feature.get(3);
			
			previous_polarity[1] = input_token[pos_parent].feature.get(5);
			
			
		}
		
		
		if (pos_parent <= 0)
		{
			prev_feature = "<START>";
		} else 
		{
			prev_feature =  this.clean(input_token[pos_parent - 1].getName());
			prev_brown_cluster = input_token[pos_parent - 1].feature.get(4);
			prev_jerboa = input_token[pos_parent - 1].feature.get(3);
			
			previous_polarity[0] = input_token[pos_parent - 1].feature.get(5);
			ther_sentiment_polarity[0] = input_token[pos_parent - 1].feature.get(6);
		}
		
		if (pos_parent >= size - 1)
		{
			next_feature = "<END>";
		} else 
		{
			next_feature =  this.clean(input_token[pos_parent + 1].getName());
			next_brown_cluster = input_token[pos_parent + 1].feature.get(4);
			next_jerboa = input_token[pos_parent + 1].feature.get(3);
			
			previous_polarity[2] = input_token[pos_parent + 1].feature.get(5);
			ther_sentiment_polarity[2] = input_token[pos_parent+1].feature.get(6);
		}
		
		
		
		
		
		if (nodetype_parent == NodeType.Span.ordinal() && nodetype_child == NodeType.Span.ordinal())
		{
			
			int segment_start, segment_end;
			boolean isEntity = false;
			
			Out = "O";
			segment_start = pos_parent;
			segment_end = segment_start;
			
			String sentiment = PolarityType.values()[polar_parent].name();

			if (subnode_parent == SubNodeType.B.ordinal() && subnode_child == SubNodeType.A.ordinal())
			{
				Out = "Entity";
				isEntity = true;
				segment_end = pos_child;
				
				
				if (polar_parent < 2)
				{
					feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_sentiment_Feature", "", ""));
				}
			}
			else if (subnode_parent == SubNodeType.B.ordinal() && subnode_child == SubNodeType.B.ordinal())
			{
				
				segment_start = pos_parent;
				segment_end = segment_start;
				
				sentiment = PolarityType.values()[polar_parent].name();
			}
			else if (subnode_parent == SubNodeType.A.ordinal() && subnode_child == SubNodeType.A.ordinal())
			{
				
				segment_start = pos_child;
				segment_end = segment_start;
				
				sentiment = PolarityType.values()[polar_child].name();
			}
			

			
			
			
			
			String word_before_segment = (segment_start - 1 < 0) ? "<START>" : input_token[segment_start - 1].getName();
			String word_after_segment = (segment_end + 1 >= size) ? "<END>" : input_token[segment_end + 1].getName();
			
			String word_first_segment = input_token[segment_start].getName();
			String word_last_segment = input_token[segment_end].getName();
			
			
			
			feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_word_before_segment:" , Out, word_before_segment));
			feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_word_after_segment:" , Out, word_after_segment));
			if (isEntity)
				feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_sentiment:" , Out, sentiment));
			
			
			//feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_prefix_3:" , Out, word_first_segment.substring(0, 3)));
			//feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_suffix_3:" , Out, word_last_segment.substring(word_last_segment.length() - 3)));
			
			String Sent = sentiment;
			String NE = Out + "_" + sentiment;
			
			for(int k = segment_start; k <= segment_end; k++)
			{
				WordWithFeatureToken token = input_token[k];
				String word = token.getName();
				String word_shape = getShape(word);
				
				int index = k - segment_start;
				int rindex = segment_end - k;
				
				feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_indexed_word:" , Out, index + "_" + word ));
				feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_rindexed_word:" , Out, rindex + "_" + word ));	
				
				feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_word:" , Out, word ));
				
				feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_indexed_wordshape:" , Out, index + "_" + word_shape ));
				feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_rindexed_wordshape:" , Out, rindex + "_" + word_shape ));		
				
				feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_wordshape:" , Out, word_shape ));
				
				feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "NE_contains" , Out, "first_letter_upper:" +  Character.isUpperCase(word_parent.charAt(0))));
				
				int pos = k;
				prefix = "word_";
				
				WordWithFeatureToken current = input_token[pos];
				
				
				for (String[] f : mynetwork.feature[pos]) {
					addPrefix(featureArr, f, "indexed_" + prefix + index, Sent, NE);
					addPrefix(featureArr, f, "rindexed_" + prefix + rindex, Sent, NE);
					addPrefix(featureArr, f, prefix, Sent, NE);
					
				}
				
				
				//int num_sent = 0;
				for (String[] f : mynetwork.feature[pos]) {
					if (f[0].indexOf("_is_sent") >= 0)
						num_sent++;
				}
	
				
			}
			
			
			for (int i = segment_start - 1; i >= segment_start - 3 && i >= 0; i--) {
				prefix = "_immediate_prev";
				for (String[] f : mynetwork.horizon_feature[i]) 
				{
					addPrefix(featureArr, f, prefix, Sent, NE);
				}
				
				prefix = "wordm" + (segment_start - i);
				for (String[] f : mynetwork.feature[i]) 
				
				{
					addPrefix(featureArr, f, prefix, Sent, NE);
				}


			}

			

			for (int i = segment_end + 1; i <= segment_end + 3 && i < size; i++) {
				prefix = "_immediate_next";
				for (String[] f : mynetwork.horizon_feature[i]) {
					addPrefix(featureArr, f, prefix, Sent, NE);
				}
				
				prefix = "wordp" + (i - segment_end );
				for (String[] f : mynetwork.feature[i]) {
					addPrefix(featureArr, f, prefix, Sent, NE);
				}


			}

			
			
			prefix = "_sent_overall";
			String[] sent_overall = null;
	
			if (num_sent == 1)
				sent_overall = new String[] {"", "", "_has_one_sent" };

			if (num_sent == 2)
				sent_overall = new String[] { "", "", "_has_two_sent"};

			if (num_sent == 3)
				sent_overall = new String[] {"", "",  "_has_three_sent" };

			if (num_sent > 3)
				sent_overall = new String[] { "", "", "_has_alotta_sent"};
			
			if (num_sent > 0)
				addPrefix(featureArr, sent_overall, prefix, Sent, NE);
	
		} 
		
		if (featureArr.size() > 0) {
			for (String[] item : featureArr) {
				//if (item[0].startsWith("word_id"))
				//System.out.println(Arrays.toString(item));
				feature.add(this._param_g.toFeature(item[0] + item[2], Out,  "_link(NE,SENT)"));
				
			}
		}
		
		
		
		
		if (subnode_parent == SubNodeType.B.ordinal() && subnode_child == SubNodeType.B.ordinal())
		{
			
			feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "O_hidden_sentiment" , PolarityType.values()[polar_parent].name()+"_Before", "current_word:" +word_parent ));
			feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "O_hidden_sentiment" , PolarityType.values()[polar_parent].name(), "current_word:" +word_parent ));
			//feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "O_hidden_sentiment" , PolarityType.values()[polar_parent].name(),   "current_word:" +word_parent + "|||" +  "next_word:" +next_feature ));
			
			//new
			if (!previous_polarity[1].equals("_") && previous_polarity[1].toLowerCase().contains(PolarityType.values()[polar_parent].name()))
			{
				feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "O_hidden_sentiment" , PolarityType.values()[polar_parent].name() ,  "current_previous_polarity:" + previous_polarity[1]));
			}
			//???
			if (!ther_sentiment_polarity[1].equals("_"))
				feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "O_hidden_sentiment" , PolarityType.values()[polar_parent].name() ,  "current_ther_sentiment_polarity:" + ther_sentiment_polarity[1]));
			
			
		}
		else if (subnode_parent == SubNodeType.A.ordinal() && subnode_child == SubNodeType.A.ordinal())
		{
			
			feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "O_hidden_sentiment" , PolarityType.values()[polar_child].name()+"_After","current_word:" + word_parent  ));
			feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "O_hidden_sentiment" , PolarityType.values()[polar_child].name(),"current_word:" + word_parent  ));
			//feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "O_hidden_sentiment" , PolarityType.values()[polar_parent].name(),   "current_word:" +word_parent + "|||" +  "next_word:" + next_feature ));
			
			
			//new
			if (!previous_polarity[1].equals("_") && previous_polarity[1].toLowerCase().contains(PolarityType.values()[polar_child].name()))
				feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "O_hidden_sentiment" , PolarityType.values()[polar_child].name() ,  "current_previous_polarity:" + previous_polarity[1]));
			//????
			if (!ther_sentiment_polarity[1].equals("_"))
				feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "O_hidden_sentiment" , PolarityType.values()[polar_child].name() ,  "current_ther_sentiment_polarity:" + ther_sentiment_polarity[1]));
			
		} else if (subnode_parent == SubNodeType.A.ordinal() && subnode_child == SubNodeType.B.ordinal())
		{
			//feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "O_hidden_sentiment_change" , PolarityType.values()[polar_child].name(), PolarityType.values()[polar_parent].name() ));
			//feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "O_hidden_sentiment_change" ,  PolarityType.values()[polar_parent].name(),PolarityType.values()[polar_child].name() ));
			//feature.add(this._param_g.toFeature(FEATURE_TYPES.Unigram + "-" + "O_hidden_sentiment_change" , PolarityType.values()[polar_parent].name() + "|||" + PolarityType.values()[polar_child].name(),  word_parent + "|||" + next_feature ));
			
		}
		
		
		
		
		
		if (feature.size() > 0)
		{
			int f[] = new int[feature.size()];
	
			for(int i = 0; i < f.length; i++)
				f[i] = feature.get(i);
	
			fa =  new FeatureArray(f);
		}
		return fa;
	}
	
	public void get_features(LinearNetwork mynetwork, WordWithFeatureToken[] input_token, int size)
	{

		mynetwork.feature = new ArrayList[size];
		mynetwork.horizon_feature = new ArrayList[size];

		for (int i = 0; i < size; i++) {

			ArrayList<String[]> featureArr = new ArrayList<String[]>();
			WordWithFeatureToken current = input_token[i];
			String word = current.feature.get(WORD_FEATURE_TYPES._id_.ordinal());
			String escape_word = word.toLowerCase();
			escape_word = this.clean(escape_word);
			
			
			String prior_polarity = current.feature.get(WORD_FEATURE_TYPES._sent_.ordinal());
			prior_polarity = clean(prior_polarity);
			
			if (!prior_polarity.endsWith("_"))
			{
				featureArr.add(new String[] {"_is_sent","","" });
			}
			
			String prior_ther_polarity = current.feature.get(WORD_FEATURE_TYPES._sent_ther_.ordinal());
			prior_ther_polarity = prior_ther_polarity.replaceAll(":", "_").toLowerCase();
			
			if (!prior_ther_polarity.endsWith("_"))
			{
				featureArr.add(new String[] {"_is_sent_ther", "","" });
			}
			
			String Out = "";

			featureArr
					.add(new String[] {
							WORD_FEATURE_TYPES._id_.toString(),
							"",escape_word
							 });
			featureArr.add(new String[] {
					WORD_FEATURE_TYPES._sent_.toString(),
					"",
					prior_polarity });
			featureArr.add(new String[] {
					WORD_FEATURE_TYPES._sent_ther_.toString(),
					"",
					prior_ther_polarity });
			featureArr.add(new String[] {
					WORD_FEATURE_TYPES.brown_clusters3.toString() + ":",
					"",
					current.feature.get(WORD_FEATURE_TYPES.brown_clusters3
							.ordinal()) });
			
			featureArr.add(new String[] {
					WORD_FEATURE_TYPES.brown_clusters5.toString() + ":",
					"",
					current.feature.get(WORD_FEATURE_TYPES.brown_clusters5
							.ordinal()) });
			
			String word_parent = word;

			if (TargetSentimentGlobal.word_feature_on) {
				
				featureArr.add(new String[] { "_NE_word_length", Out,
						word_parent.length() + "" });

				String sentence_pos = "";
				if (i == 0)
					sentence_pos = "is_first";
				else if (i == 1)
					sentence_pos = "is_second";
				else if (i == 2)
					sentence_pos = "is_third";
				else if (i == size - 1)
					sentence_pos = "is_last";
				else if (i == size - 2)
					sentence_pos = "is_second_last";
				else if (i == size - 3)
					sentence_pos = "is_third_last";
				else
					sentence_pos = "in_middle";

				featureArr.add(new String[] { "_NE_sentence_pos:", Out,
						sentence_pos });

				String message_len_group = "";
				if (size <= 5)
					message_len_group = "1";
				else if (size <= 10)
					message_len_group = "2";
				else
					message_len_group = "3";

				featureArr.add(new String[] { "_NE_message_len_group:", Out,
						message_len_group });

				if (word_parent.length() == 3 || word_parent.length() == 4)
					featureArr.add(new String[] { "_NE_3_4_letters", Out, "" });

				if (Character.isUpperCase(word_parent.charAt(0)))
					featureArr.add(new String[] { "_NE_first_letter_upper",
							Out, "" });
				
				int uppercase_counter = 0;
				boolean hasDash = false, hasDigit = false, allLowercase = true;
				String punc = "";
				char prev = 0;
				int repeat = 0;
				for (int j = 0; j < word_parent.length(); j++) {
					char ch = word_parent.charAt(j);
					if (Character.isUpperCase(ch))
						uppercase_counter++;

					if (ch == '-')
						hasDash = true;

					if (Character.isDigit(ch))
						hasDigit = true;

					if (ch == prev)
						repeat++;

					prev = ch;
				}
				if (uppercase_counter > 1) {
					allLowercase = false;
					featureArr.add(new String[] {
							"_NE_more_than_one_letter_upper", Out, "" });
				}

				if (hasDash)
					featureArr.add(new String[] { "_NE_has_dash", Out, "" });

				if (hasDigit)
					featureArr.add(new String[] { "_NE_has_digit", Out, "" });

				if (allLowercase)
					featureArr
							.add(new String[] { "_NE_is_lower_case", Out, "" });

				if (repeat > 1)
					featureArr.add(new String[] { "_sent_has_repeat", Out,
							"" });

				if (word_parent.indexOf('!') >= 0)
					featureArr.add(new String[] { "_sent_has_exclaim", Out,
							"" });

				if (word_parent.indexOf("!!") >= 0)
					featureArr.add(new String[] { "_sent_has_many_exclaim",
							Out, "" });

				if (word_parent.indexOf("...") >= 0)
					featureArr.add(new String[] { "_sent_has_ellipse", Out,
							"" });

				if (word_parent.indexOf("?") >= 0)
					featureArr.add(new String[] { "_sent_has_question",
							Out, "" });

				if (word_parent.indexOf("??") >= 0)
					featureArr.add(new String[] {
							"_sent_has_many_question", Out, "" });

				String lowercase = word_parent.toLowerCase();
				String[] laugh = new String[] { "haha", "jaja", "jeje",	"hehe", "hihi", "jiji" };
				for (int j = 0; j < laugh.length; j++)
					if (lowercase.indexOf(laugh[j]) >= 0) {
						featureArr.add(new String[] { "_sent_has_laugh", Out, "" });
						break;
					}

				
				String jerboa = current.feature.get(WORD_FEATURE_TYPES.jerboa.ordinal()).toLowerCase();
				
				if (jerboa.indexOf("emoticon") >= 0 || jerboa.indexOf("smiling") >= 0 || jerboa.indexOf("frowning") >= 0)
				{
					if (word_parent.indexOf(')') >= 0 || word_parent.indexOf('D') >= 0 	|| word_parent.indexOf(']') >= 0)
						featureArr.add(new String[] {"_sent_jerb_happy", Out, "" });

					if (word_parent.indexOf('(') >= 0 || word_parent.indexOf('[') >= 0) 
						featureArr.add(new String[] { "_sent_jerb_sad", Out, "" });

				}

				for (String filename : TargetSentimentGlobal.LinguisticFeaturesLibaryName) {
					if (filename
							.equals(TargetSentimentGlobal.LinguisticFeaturesLibaryNamePart[0]))
						continue;

					ArrayList<String> libarylist = LinguisticFeaturesLibaryList
							.get(filename);
					if (libarylist == null)
						continue;
					
					String featurename =  LinguisticFeatureLibNameFeatureMapping.get(filename);
					if (featurename == null)
					{
						featurename = filename;
						if (featurename.endsWith(".txt"))
						{
							featurename = featurename.substring(0, featurename.length() - 4);
						}
					}

					if (libarylist.indexOf(word_parent) >= 0) {
						featureArr.add(new String[] { "_sent_" + featurename,
								"", "" });
						continue;

					}
				}

				String filename = TargetSentimentGlobal.LinguisticFeaturesLibaryNamePart[0];
				ArrayList<String> libarylist = LinguisticFeaturesLibaryList.get(filename);
				String featurename = LinguisticFeatureLibNameFeatureMapping.get(filename);
				if (featurename == null)
				{
					featurename = filename;
					if (featurename.endsWith(".txt"))
					{
						featurename = featurename.substring(0, featurename.length() - 4);
					}
				}
				
				for (String prefix : libarylist)
					if (word_parent.indexOf(prefix) >= 0) {
						featureArr
								.add(new String[] {
										"_sent_"
												+ featurename,
										"", "" });
					}
				
				
				if (escape_word.equals("my") || escape_word.equals("mis") || escape_word.equals("mi"))
				{
					featureArr.add(new String[]{"_sent_has_mi", "", ""});
				}
		        if (escape_word.equals("nunca") || escape_word.equals("nadie"))
		        {
		        	featureArr.add(new String[]{"_sent_has_no_word", "", ""});
		            
		        }
			}

			mynetwork.feature[i] = featureArr;
			mynetwork.horizon_feature[i] = new ArrayList<String[]>();
			
			for(String[] item : featureArr)
			{
				if (item[0].contains("sent_"))
				{
					mynetwork.horizon_feature[i].add(item);
				}
			}
		}

		
	}
	
	public void addPrefix(ArrayList<String[]> featureArr, String[] f, String prefix, String Sent, String NE)
	{
		
		featureArr.add(new String[] { prefix + f[0], f[1], f[2]});
	}
	
	
	
	public String escape(String s)
	{
		for(Character val : string_map.keySet())
		{
			String target = val + "";
			if (s.indexOf(target) >= 0)
			{
				String repl = string_map.get(val);
				s = s.replace(target, repl);
				
			}
		}
		
		return s;
	}
	
	public String norm_digits(String s)
	{
		s = s.replaceAll("\\d+", "0");
		
		return s;
		
	}
	public String clean(String s)
	{
		if (s.startsWith("http://") || s.startsWith("https://"))
		{
			s = "<WEBLINK>";
		}
		else
		{
			s = norm_digits(s.toLowerCase());
			s = escape(s);
			s = s.replaceAll("[^A-Za-z0-9_]", "_");
		}
		
		return s;
	}
	
	public String getShape(String s)
	{
		String shape = s;
		if (shape.startsWith("http://") || shape.startsWith("https://"))
		{
			shape = "<WEBLINK>";
		}
		else
		{
			shape = norm_digits(s.toLowerCase());
			
			shape = shape.replaceAll("[A-Z]", "X");
			shape = shape.replaceAll("[a-z]", "x");
			
			
		}
		
		return shape;
	}
	
	
	public static final HashMap<Character , String> string_map = new HashMap<Character , String>(){
		{
			put('.', "_P_");	put(',', "_C_");
			put('\'', "_A_"); 	put('%', "_PCT_");
			put('-', "_DASH_");	put('$', "_DOL_");
			put('&', "_AMP_");	put(':', "_COL_");
			put(';', "_SCOL_");	put('\\', "_BSL_");
			put('/', "_SL_");	put('`',"_QT_");
			put('?',"_Q_");		put('¿',"_QQ_");
			put('=', "_EQ_");	put('*', "_ST_");
			put('!', "_E_");	put('¡', "_EE_");
			put('#', "_HSH_");	put('@', "_AT_");
			put('(', "_LBR_");	put(')', "_RBR_"); 
	        put('\"', "_QT0_"); put('Á',"_A_ACNT_"); 
	        put('É',"_E_ACNT_");put('Í',"_I_ACNT_");
            put('Ó',"_O_ACNT_");put('Ú',"_U_ACNT_"); put('Ü',"_U_ACNT0_"); put('Ñ',"_N_ACNT_");
            put('á',"_a_ACNT_");put('é',"_e_ACNT_"); put('í',"_i_ACNT_"); put('ó',"_o_ACNT_");
            put('ú',"_u_ACNT_");put('ü',"_u_ACNT0_"); put('ñ',"_n_ACNT_"); put('º',"_deg_ACNT_");
		}
	};
	


	public static final HashMap<String , String> LinguisticFeatureLibNameFeatureMapping = new HashMap<String , String>(){
	{
		put("Neg_Prefix_Eng.txt", "Neg_Prefix_Eng");
		put("Neg_Prefix_Sp.txt", "Neg_Prefix_Sp");
		
		put("bad_words","_sent_has_mal");
		put("good_words", "_sent_has_bien");
		put("curse_words", "_sent_has_curse_word");
		put("determiners", "_is_determiner");
		put("prepositions", "_is_preposition");
		
		
	}
};
	
	
	
	

}



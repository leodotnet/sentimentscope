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

import java.util.Random;

public class NetworkConfig {
	
	public static Random r = new Random();
	public static double FEATURE_INIT_WEIGHT = 0;//r.nextDouble();//Math.log(1E-10);//Math.log(1);
	public static boolean RANDOM_INIT_WEIGHT = true;
	public static int RANDOM_INIT_FEATURE_SEED = 1234;
	
	public static boolean CACHE_FEATURES_AT_CONJUNCTIVE_CELLS = true;
	public static int PRUNE_MIN_LENGTH = 1000;
	public static double PRUNE_THRESHOLD = 1000;
	public static boolean TRAIN_MODE_IS_GENERATIVE = true;
	public static boolean CACHE_FEATURE_SCORES = false;
	public static boolean diagco = false;
	public static int[] iprint = {0,0};
	public static double eps = 10e-3;
	public static double xtol = 10e-16;
	public static int[] iflag = {0};
	public static double L2_REGULARIZATION_CONSTANT = 0.01;
	public static int _FOREST_MAX_HEIGHT = 10000;
	public static int _FOREST_MAX_WIDTH = 10000;
	public static int _NETWORK_MAX_DEPTH = 901;
	public static int _nGRAM = 1;//2;//1;
	public static double objtol = 10e-15;//the value used for checking the objective increment for generative models.
	
	public static int _SEMANTIC_FOREST_MAX_DEPTH = 20;//the max depth of the forest when creating the semantic forest.
	public static int _SEMANTIC_PARSING_NGRAM = 1;//2;
	
	public static boolean DEBUG_MODE = false;//true;//false;//true;
	public static boolean REBUILD_FOREST_EVERY_TIME = false;
	
	public static boolean _CACHE_FEATURES_DURING_TRAINING = true;
	
	
	public static int _numThreads = 10;
	
	public static int _maxSpanLen = 2;//the upper-bound of the length of a span.
	
	public static boolean ENABLE_MAX_MARGINAL = false;
	
}
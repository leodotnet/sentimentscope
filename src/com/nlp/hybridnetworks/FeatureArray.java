/** Statistical Natural Language Processing System
    Copyright (C) 2014-2015  Lu, Wei

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

import com.nlp.targetedsentiment.f.latent.TargetSentimentFeatureManager;

public class FeatureArray implements Serializable{
	
	private static final long serialVersionUID = 9170537017171193020L;
	
	private double _score;
	private int[] _fs;
	private boolean _isLocal = false;
	private FeatureArray _next;
	private double[] _fv;
	
	public static FeatureArray EMPTY = new FeatureArray(new int[0]);
	public static FeatureArray NEGATIVE_INFINITY = new FeatureArray(-10000);
//	public static FeatureArray NEGATIVE_INFINITY = new FeatureArray(Double.NEGATIVE_INFINITY);
	public static boolean ENABLE_WORD_EMBEDDING = false;

	public String viewCurrent(){
		return Arrays.toString(this._fs);
	}

	public FeatureArray(int[] fs, FeatureArray next) {
		this._fs = fs;
		this._next = next;
	}

	public FeatureArray(int[] fs) {
		this._fs = fs;
		this._next = null;
	}
	
	//word embedding
	public FeatureArray(int[] fs, double[] fv)
	{
		this._fs = fs;
		this._fv = fv;
		this._next = null;
	}
	
	public FeatureArray(double score) {
		this._score = score;
	}
	
	public FeatureArray toLocal(LocalNetworkParam param){
		if(this==NEGATIVE_INFINITY){
			return this;
		}
		if(this._isLocal){
			return this;
		}
		
		int[] fs_local = new int[this._fs.length];
		for(int k = 0; k<this._fs.length; k++){
			fs_local[k] = param.toLocalFeature(this._fs[k]);
			if(fs_local[k]==-1){
				throw new RuntimeException("The local feature got an id of -1 for "+this._fs[k]);
			}
		}
		
		FeatureArray fa;
		if(this._next!=null){
			fa = new FeatureArray(fs_local, this._next.toLocal(param));
		} else {
			if (ENABLE_WORD_EMBEDDING)
			{
				fa = new FeatureArray(fs_local, _fv);
			}
			else
			{
				fa = new FeatureArray(fs_local);
			}
			
		}
		fa._isLocal = true;
		return fa;
	}
	
	public int[] getCurrent(){
		return this._fs;
	}
	
	public FeatureArray getNext(){
		return this._next;
	}
	
	public void update(LocalNetworkParam param, double count){
		if(this == NEGATIVE_INFINITY){
			return;
		}
		
//		if(!this._isLocal)
//			throw new RuntimeException("This feature array is not local");
		
		int[] fs_local = this.getCurrent();
		if (ENABLE_WORD_EMBEDDING)
		{
			for(int i = 0; i < fs_local.length; i++)
			{
				int f_local = fs_local[i];
				param.addCount(f_local, count * _fv[i]);

			}
			
		}
		else
		{
			for(int f_local : fs_local){		
				param.addCount(f_local, count);

			}
		}
		
		if(this._next!=null){
			this._next.update(param, count);
		}
		
	}
	
	public double getScore(LocalNetworkParam param){
		if(this == NEGATIVE_INFINITY){
			return this._score;
		}
		if(!this._isLocal != param.isGlobalMode()) {
			throw new RuntimeException("This FeatureArray is local? "+this._isLocal+"; The param is "+param.isGlobalMode());
		}
		
		//if the score is negative infinity, it means disabled.
		if(this._score == Double.NEGATIVE_INFINITY){
			return this._score;
		}
		
		this._score = this.computeScore(param, this.getCurrent());
		
		if(this._next!=null){
			this._score += this._next.getScore(param);
		}
		
		return this._score;
	}
	
	private double computeScore(LocalNetworkParam param, int[] fs){
		if(!this._isLocal != param.isGlobalMode()) {
			throw new RuntimeException("This FeatureArray is local? "+this._isLocal+"; The param is "+param.isGlobalMode());
		}
		
		double score = 0.0;
		if (ENABLE_WORD_EMBEDDING)
		{
			
			for(int i = 0; i < fs.length; i++)
			{
				int f = fs[i];
				if (f != -1)
				{
					//System.out.println(_fv.length + "");
					score += param.getWeight(f) * _fv[i];
				}
			}
			
		
		}
		else
		{
			for(int f : fs){
				if(f!=-1){
					score += param.getWeight(f);
				}	
			}
		}
		return score;
	}
	
	//returns the number of elements in the feature array
	public int size(){
		int size = this._fs.length;
		if(this._next!=null){
			size += this._next.size();
		}
		return size;
	}
	
}
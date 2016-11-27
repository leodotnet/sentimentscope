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
package com.nlp.commons.ml.opt;

import java.util.Arrays;

import com.nlp.commons.ml.opt.LBFGS.ExceptionWithIflag;

public abstract class Optimizer {

	private int _n;
	private int _m = 4;
	private int[] _iprint = {0,0};
	private int[] _iflag = {0};
	private double _eps = 10e-10;
	private double _xtol = 10e-16;
	private double _f;
	private double[] _diag;
	protected double[] _x;
	protected double[] _g;
	private boolean _diagco = false;
	
	
	
	public void setObjective(double f){
		this._f = f;
	}

	public void setVariables(double[] x){
		this._x = x;
		this._n = x.length;
		if(this._diag == null)
		{
			this._diag = new double[this._n];
			for(int k = 0; k<this._n; k++)
				this._diag[k] = 1.0;
		}
	}
	
	public void setGradients(double[] g){
		this._g = g;
	}
	
	public void updateVariables(double[] x){
		this._x = x;
	}
	
	//return true if it should stop.
	public abstract boolean optimize() throws ExceptionWithIflag;
    	
	
	
    private static double square(double x){
    	return x * x;
    }
    
    //v : dim = 3
    private static double getObj(double[] v, double factor){
    	double numerator = Math.exp(v[0]) + Math.exp(v[1]);
    	double denominator = Math.exp(v[0]) + Math.exp(v[1]) + Math.exp(v[2]);
    	double result = numerator/denominator * factor;
    	return -result;
    }

    private static double getLogObj(double[] v){
    	double numerator1 = Math.exp(v[0]) + Math.exp(v[1]);
//    	double numerator2 = Math.exp(v[1]) + Math.exp(v[2]);
    	double denominator1 = Math.exp(v[0]) + Math.exp(v[1]) + Math.exp(v[2]);
//    	double denominator2 = Math.exp(v[0]) + Math.exp(v[1]) + Math.exp(v[2]);
    	double result = Math.log(numerator1) - Math.log(denominator1);
//    			+ Math.log(numerator2) - Math.log(denominator2);
    	return -(result);
    }
    
    private static double[] getGradient(double[] v, double factor){
    	double g[] = new double[v.length];
    	double denominator = Math.exp(v[0]) + Math.exp(v[1]) + Math.exp(v[2]);
    	g[0] = -Math.exp(v[0]+v[2])/Math.pow(denominator, 2) * factor;
    	g[1] = -Math.exp(v[1]+v[2])/Math.pow(denominator, 2) * factor;
    	g[2] = Math.exp(v[2])*(Math.exp(v[0]) + Math.exp(v[1]))/Math.pow(denominator, 2) * factor;
    	return g;
    }
    
    private static double[] getLogGradient(double[] v){
    	double g[] = new double[v.length];
    	double denominator = Math.exp(v[0]) + Math.exp(v[1]) + Math.exp(v[2]);
    	g[0] = -Math.exp(v[0]+v[2])/denominator/(Math.exp(v[0]) + Math.exp(v[1]));
    	g[1] = -Math.exp(v[1]+v[2])/denominator/(Math.exp(v[0]) + Math.exp(v[1]));
    	g[2] = Math.exp(v[2])/denominator;
    	return g;
    }
    

    private static double[] getGradient_alternative(double[] v){
    	double g[] = getGradient(v,1.0);
    	
    	double obj = getObj(v, 1.0);
    	double g_p[] = getLogGradient(v);
    	
    	for(int k = 0; k<g_p.length; k++){
    		g_p[k] *= obj;
    	}
    	
    	System.err.println(Arrays.toString(g));
    	System.err.println(Arrays.toString(g_p));
    	System.exit(1);
    	
    	return g_p;
    }
    
    private static double[] getGradient_approx(double[] v){
    	double g_p[] = getGradient(v,1);
    	
    	double g[] = new double[v.length];
    	double v1[] = (double[])v.clone();
    	double step = 0.001;
    	double f_diff;
    	
    	v1[0] += step;
    	f_diff = getObj(v1,1) - getObj(v,1);
    	g[0] = f_diff / step;
    	v1[0] -= step;
    	
    	v1[1] += step;
    	f_diff = getObj(v1,1) - getObj(v,1);
    	g[1] = f_diff / step;
    	v1[1] -= step;
    	
    	v1[2] += step;
    	f_diff = getObj(v1,1) - getObj(v,1);
    	g[2] = f_diff / step;
    	v1[2] -= step;
    	
    	System.err.println(Arrays.toString(g_p));
    	System.err.println(Arrays.toString(g));
    	System.exit(1);
    	
    	return g;
    }
    
    private static double[] getLogGradient_approx(double[] v){
    	double g_p[] = getLogGradient(v);
    	
    	double g[] = new double[v.length];
    	double v1[] = (double[])v.clone();
    	double step = 0.001;
    	double f_diff;
    	
    	v1[0] += step;
    	f_diff = getLogObj(v1) - getLogObj(v);
    	g[0] = f_diff / step;
    	v1[0] -= step;
    	
    	v1[1] += step;
    	f_diff = getLogObj(v1) - getLogObj(v);
    	g[1] = f_diff / step;
    	v1[1] -= step;
    	
    	v1[2] += step;
    	f_diff = getLogObj(v1) - getLogObj(v);
    	g[2] = f_diff / step;
    	v1[2] -= step;
    	
    	System.err.println(Arrays.toString(g_p));
    	System.err.println(Arrays.toString(g));
    	System.exit(1);
    	
    	return g;
    }
    
    
    
}

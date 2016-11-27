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

public class LBFGSOptimizer2 {

	private int _n;
	private int _m = 4;
	private int[] _iprint = {0,0};
	private int[] _iflag = {0};
	private double _eps = 10e-10;
	private double _xtol = 10e-16;
	private double _f;
	private double[] _diag;
	private double[] _x;
	private double[] _g;
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
	public boolean optimize() throws ExceptionWithIflag{
    	LBFGS.lbfgs(this._n, this._m, this._x, this._f, this._g, this._diagco, this._diag, this._iprint, this._eps, this._xtol, this._iflag);
    	return _iflag[0] == 0;
	}
	
    private static double square(double x){
    	return x * x;
    }
    
    //v : dim = 3
    private static double getObj(double[] v){
    	double r = 0;
    	r += Math.pow(v[0], 4);
    	r += -2*Math.pow(v[0], 3);
    	r += Math.pow(v[0], 2);
    	r += 10;
    	return r;
    }
    
    private static double getLogObj(double[] v){
    	double r = getObj(v);
    	r = Math.log(r);
    	return r;
    }
    
    //x^4-2x^3+x^2+10
    private static double[] getGradient(double[] v){
    	double[] g = new double[v.length];
    	double r = 0;
    	r += 4 * Math.pow(v[0], 3);
    	r += -6 * Math.pow(v[0], 2);
    	r += 2 * v[0];
    	g[0] = r;
    	return g;
    }
    
    private static double[] getLogGradient(double[] v){
    	double g[] = new double[v.length];
    	double r = 0;
    	r = 2 * v[0] * (2 * Math.pow(v[0], 2) - 3 * v[0] + 1);
    	r /= (Math.pow(v[0], 4) - 2 * Math.pow(v[0], 3) + Math.pow(v[0], 2) + 10);
    	g[0] = r;
    	return g;
    }
    

    private static double[] getGradient_alternative(double[] v){
    	double g[] = getGradient(v);
    	
    	double obj = getObj(v);
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
    	double g_p[] = getGradient(v);
    	
    	double g[] = new double[v.length];
    	double v1[] = (double[])v.clone();
    	double step = 1E-8;
    	double f_diff;
    	
    	v1[0] += step;
    	f_diff = getObj(v1) - getObj(v);
    	g[0] = f_diff / step;
    	v1[0] -= step;
    	
    	v1[1] += step;
    	f_diff = getObj(v1) - getObj(v);
    	g[1] = f_diff / step;
    	v1[1] -= step;
    	
    	v1[2] += step;
    	f_diff = getObj(v1) - getObj(v);
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
    
    public static void main(String args[]) throws ExceptionWithIflag{
    	
    	LBFGSOptimizer2 opt = new LBFGSOptimizer2();
    	
//    	double lambda = Math.abs(Math.random()*10);
//    	System.err.println("lambda="+lambda);
    	
		double f; 
    	
    	double[] x;
    	
    	x = new double[]{(Math.random())*100};
    	
//    	getGradient_approx(x);
//    	getLogGradient_approx(x);
//    	getGradient_alternative(x);
    	
//    	x = new double[]{-100,-100,100};
    	x = new double[]{0.1};
    	System.err.println(x[0]);
    	opt.setVariables(x);
    	
    	while(true){
        	f = getLogObj(x);
        	double[] g = getLogGradient(x);
        	opt.setObjective(f);
        	opt.setGradients(g);
        	boolean done = opt.optimize();
        	if(done) break;
        	System.err.println("Log Obj="+getLogObj(x));
        	System.err.println("g:"+Arrays.toString(g));
        	System.err.println("x:"+Arrays.toString(x));
    	}
    	
    	double obj_old = getLogObj(x);
    	double old_x[] = x.clone();
    	
    	double factor = 1/getLogObj(x);
    	
    	if(Double.isInfinite(factor)){
    		System.err.println("DONE. REACHED GLOBAL MIN");
    		System.exit(1);
    	}
    	
//    	System.err.println("F=="+getLogObj(new double[]{95306.97791649547}));
    	
    	System.exit(1);
    	
    	System.err.println("FACTOR="+factor);
    	System.err.println("===========");
    	System.err.println("===========");
    	System.err.println("===========");
    	System.err.println("===========");
    	
//    	double factor = 1E10;
    	
    	while(true){
        	f = getObj(x);
        	double[] g = getGradient(x);
        	System.err.println("Obj="+getObj(x));
        	System.err.println("g:"+Arrays.toString(g));
        	System.err.println("x:"+Arrays.toString(x));
        	opt.setObjective(f);
        	opt.setGradients(g);
        	boolean done = opt.optimize();
        	if(done) break;
        	System.err.println("g:"+Arrays.toString(g));
        	System.err.println("x:"+Arrays.toString(x));
    	}
    	
    	System.err.println("+++++++++++");
    	System.err.println("+++++++++++");
    	System.err.println("+++++++++++");
    	System.err.println("+++++++++++");
    	
    	while(true){
        	f = getLogObj(x);
        	double[] g = getLogGradient(x);
        	opt.setObjective(f);
        	opt.setGradients(g);
        	boolean done = opt.optimize();
        	if(done) break;
        	System.err.println("Log Obj="+getLogObj(x));
        	System.err.println("g:"+Arrays.toString(g));
        	System.err.println("x:"+Arrays.toString(x));
    	}
    	
    	double[] g = getGradient(x);
    	System.err.println("g:"+Arrays.toString(g));
    	System.err.println("x:\t"+Arrays.toString(x));
    	System.err.println("old x:\t"+Arrays.toString(old_x));
    	
    	System.err.println("Obj="+getObj(x));
    	System.err.println("Log Obj="+getLogObj(x));
    	System.err.println("Old Obj="+obj_old);
    }
	
    public static void main2(String args[]) throws ExceptionWithIflag{
    	
    	LBFGSOptimizer2 opt = new LBFGSOptimizer2();
    	
    	double lambda = Math.abs(Math.random()*10);
    	System.err.println("lambda="+lambda);
    	
		double v; 
    	double f; 
    	
    	double[] x;
    	
    	double A = (Math.random()-.5);
    	double B = (Math.random()-.5);
    	
    	x = new double[]{(Math.random()-.5)*100, (Math.random()-.5)*100};
    	System.err.println(x[0]+".."+x[1]);
    	opt.setVariables(x);
    	
    	while(true){
    		v = Math.exp(A*x[0]+B*x[1]);
        	f = -1.0/(1+v)+lambda*(square(x[0])+square(x[1])+square(x[0]-x[1]))-100*(x[0]);
        	double[] g = {A*v/square(1+v)+2*lambda*x[0]+2*lambda*(x[0]-x[1])-100,B*v/square(1+v)+2*lambda*x[1]+2*lambda*(x[1]-x[0])};
        	opt.setObjective(f);
        	opt.setGradients(g);
        	boolean done = opt.optimize();
        	if(done) break;
    	}
    	
		v = Math.exp(A*x[0]+B*x[1]);
    	double f1 = -1.0/(1+v)+lambda*(square(x[0])+square(x[1])+square(x[0]-x[1]));
    	
    	for(int i = 0; i<x.length; i++)
    		System.err.println("x["+i+"]="+x[i]);
    	
    	double a = x[0];
    	double b = x[1];
    	System.err.println("f1="+f1);
    	
    	lambda = lambda*3;
    	
    	x = new double[]{(Math.random()-.5)*100, (Math.random()-.5)*100, (Math.random()-.5)*100};
    	System.err.println(x[0]+".."+x[1]+".."+x[2]);
    	opt.setVariables(x);
    	
    	while(true){
    		double x0p = x[0]+x[2];
    		double x1p = x[1]+x[2];
    		v = Math.exp(A*x0p+B*x1p);
        	f = -1.0/(1+v)+lambda*(square(x[0])+square(x[1])+square(x[2]))-100*x[0]-100*x[2];
        	double[] g = {A*v/square(1+v)+2*lambda*x[0]-100,B*v/square(1+v)+2*lambda*x[1],(A+B)*v/square(1+v)+2*lambda*x[2]-100};
        	opt.setObjective(f);
        	opt.setGradients(g);
        	boolean done = opt.optimize();
        	if(done) break;
    	}
    	
		double x0p = x[0]+x[2];
		double x1p = x[1]+x[2];
		v = Math.exp(A*x0p+B*x1p);
    	double f2 = -1.0/(1+v)+lambda*(square(x[0])+square(x[1])+square(x[2]));
    	
    	for(int i = 0; i<x.length; i++){
    		System.err.println("x["+i+"]="+x[i]);
    	}
    	System.err.println("f2="+f2);
    	
    	System.err.println("DIFF:");
    	System.err.println(x0p-a);
    	System.err.println(x1p-b);
    	System.err.println(x[0]+x[1]-x[2]);
    	System.err.println(f2-f1);
    	
    }
    
}

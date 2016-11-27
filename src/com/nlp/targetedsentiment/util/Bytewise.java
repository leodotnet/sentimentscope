package com.nlp.targetedsentiment.util;

public class Bytewise {

	public static double ByteToDouble(byte[] a_byte)
	{
		long value = 0;
		for(int i = 0; i < a_byte.length; i++)
		{
			value = value * 256;
			value += unsignedToBytes(a_byte[i]);
		}
		
		return Double.longBitsToDouble(value);
	}
	
	public static byte[] DoubleToByte(double myDouble)
	{
		byte[] data = new byte[8];
		int pos = 0;
		
		long v = Double.doubleToLongBits(myDouble);
		
		data[pos+7] = (byte)(v);
		data[pos+6] = (byte)(v>>8);
		data[pos+5] = (byte)(v>>16);
		data[pos+4] = (byte)(v>>24);
		data[pos+3] = (byte)(v>>32);
		data[pos+2] = (byte)(v>>40);
		data[pos+1] = (byte)(v>>48);
		data[pos]   = (byte)(v>>56);
		
		return data;
	}
	
	public static int unsignedToBytes(byte b)
	{
		return b & 0xFF;
	}
	
	public static String printByte(byte[] a_byte)
	{
		String ret = "";
		ret += "{\n";
		for(int i = 0; i < a_byte.length; i++)
			ret += "  " + i + " : " + unsignedToBytes(a_byte[i]) + "\n";
		ret += "}\n";
		
		return ret;
	}
	
	public static String printDoubleToByte(double myDouble)
	{
		byte[] a_byte = DoubleToByte(myDouble);
		String ret = printByte(a_byte);
		return ret;
	}

}

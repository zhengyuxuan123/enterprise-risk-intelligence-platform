package com.example.riskplatform.common;
public class BusinessException extends RuntimeException{private final int code;public BusinessException(String m){this(400,m);}public BusinessException(int c,String m){super(m);code=c;}public int getCode(){return code;}}

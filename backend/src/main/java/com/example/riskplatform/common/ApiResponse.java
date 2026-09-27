package com.example.riskplatform.common;
import lombok.*;
@Data @NoArgsConstructor @AllArgsConstructor
public class ApiResponse<T>{private int code;private String message;private T data;public static<T>ApiResponse<T> ok(T d){return new ApiResponse<>(0,"success",d);}public static ApiResponse<Void> ok(){return new ApiResponse<>(0,"success",null);}public static<T>ApiResponse<T> fail(int c,String m){return new ApiResponse<>(c,m,null);}}
